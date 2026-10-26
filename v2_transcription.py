#!/usr/bin/env python3
"""
Transcription d'un entretien avec evaluation contre une transcription de reference
(version 6 : affichage reduit aux barres de progression + duree totale)

Principe :
  - Le programme effectue TOUJOURS trois tentatives, avec deux passes par tentative.
  - Chaque passe est comparee au fichier de transcription de reference fourni par
    --reference. Le score principal est base sur le taux d'erreur par mot (WER).
  - La transcription finale est celle dont le score face a la reference est le plus
    eleve. La reference sert uniquement a evaluer et choisir : elle ne sert PAS a
    corriger, completer ou guider les phrases generees.
  - Les passes different par : audio brut / normalise, sensibilite de la detection de
    voix, finesse du decodage, graine aleatoire, et un 2e modele si --modele2 est donne.

Affichage :
  - A l'ecran : une barre "Avancement" (preparation, puis chacune des 6 passes) et,
    en dessous, une barre des ETAPES de la passe en cours (diarisation, transcription,
    alignement, corrections, relecture). Rien d'autre pendant le calcul.
  - Tout le detail (messages des etapes, avertissements des bibliotheques, textes
    corriges...) est ecrit dans SORTIE_journal.txt. Option --verbeux : tout afficher.
  - A la fin : resultat, scores de chaque passe, DUREE TOTALE.

Sorties :
  SORTIE.csv                  resultat retenu
  SORTIE_a_verifier.csv       lignes douteuses du resultat retenu
  SORTIE_divergences.csv      passages ou les deux passes retenues different
  SORTIE_scores_reference.csv comparaison de chaque passe a la vraie transcription
  SORTIE_comparaison_modeles.csv  moyenne par modele
  SORTIE_journal.txt          journal detaille
  SORTIE_essais/              chaque passe (CSV + JSON, reprise possible) et rapport.csv

Usage :
  export HF_TOKEN=hf_xxx
  python transcription.py entretien.m4a sortie.csv \
      --reference vraie_transcription.csv --langue fr \
      --noms SPEAKER_00=Sujet SPEAKER_01=Psy \
      --modele ~/models/frwhisper-ct2 --modele2 large-v3
"""

import argparse
import bisect
import contextlib
import csv
import difflib
import gc
import itertools
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
import unicodedata
import warnings

import numpy as np
import torch
import whisperx
from tqdm import tqdm

# --------------------------------------------------------------------------
# Affichage : barres a l'ecran, details dans le journal
# --------------------------------------------------------------------------
VERBEUX = False          # --verbeux : tout afficher a l'ecran
JOURNAL = None           # chemin du fichier journal
ECRAN = sys.stdout       # flux de l'ecran (copie du descripteur d'origine, voir initialiser_affichage)


def initialiser_affichage(verbeux, chemin_journal):
    """ECRAN est une COPIE du descripteur de sortie d'origine : les barres restent
    visibles meme quand les sorties des bibliotheques sont redirigees vers le journal."""
    global VERBEUX, JOURNAL, ECRAN
    VERBEUX, JOURNAL = verbeux, chemin_journal
    sys.stdout.flush()
    ECRAN = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8", errors="replace")
    with open(JOURNAL, "a", encoding="utf-8") as fh:
        fh.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
    if not verbeux:
        warnings.filterwarnings("ignore")
        logging.disable(logging.CRITICAL)
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


def info(message=""):
    """Detail interne : ecrit dans le journal ; affiche seulement avec --verbeux."""
    if JOURNAL:
        with open(JOURNAL, "a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%H:%M:%S')} {message}\n")
    if VERBEUX:
        tqdm.write(message, file=ECRAN)


def afficher(message=""):
    """Message toujours visible (resultat final, echec d'une passe), et note au journal."""
    if JOURNAL:
        with open(JOURNAL, "a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%H:%M:%S')} {message}\n")
    tqdm.write(message, file=ECRAN)


@contextlib.contextmanager
def silencieux():
    """Redirige la sortie standard ET d'erreur (y compris celle des bibliotheques en C)
    vers le journal pendant un calcul. Les barres, ecrites sur ECRAN, restent visibles."""
    if VERBEUX or not JOURNAL:
        yield
        return
    sys.stdout.flush()
    sys.stderr.flush()
    sauve_out, sauve_err = os.dup(1), os.dup(2)
    journal = os.open(JOURNAL, os.O_WRONLY | os.O_APPEND | os.O_CREAT)
    try:
        os.dup2(journal, 1)
        os.dup2(journal, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(sauve_out, 1)
        os.dup2(sauve_err, 2)
        for fd in (sauve_out, sauve_err, journal):
            os.close(fd)


def barre_interne(*args, **kwargs):
    """Barre des sous-etapes : masquee par defaut."""
    kwargs.setdefault("file", ECRAN)
    kwargs["disable"] = not VERBEUX
    return tqdm(*args, **kwargs)


def _etape(etapes, profil, nom):
    if etapes is not None:
        etapes.set_description(f"   {profil['nom']} - {nom}")
        etapes.refresh()


def _fin_etape(etapes):
    if etapes is not None:
        etapes.update(1)


def formater_duree(secondes):
    secondes = int(round(secondes))
    h, reste = divmod(secondes, 3600)
    m, s = divmod(reste, 60)
    if h:
        return f"{h} h {m:02d} min {s:02d} s"
    if m:
        return f"{m} min {s:02d} s"
    return f"{s} s"


PROMPT_VERBATIM = "Euh, hum, hmm, mmh, ouais, ok, d'accord, ah, oh, bah, ben, voila."

INTERJECTIONS = {
    "hum", "hmm", "hm", "mm", "mmh", "mhm", "mh", "hum-hum", "mm-hm",
    "ok", "okay", "okey", "d'accord", "ah", "oh", "ouais", "mouais",
}

HALLUCINATIONS = ("sous-titr", "amara", "merci d'avoir regard", "abonnez",
                  "radio-canada")

SR = 16000


# --------------------------------------------------------------------------
# Utilitaires
# --------------------------------------------------------------------------
def extraire_audio(video, wav):
    subprocess.run(
        ["ffmpeg", "-y", "-i", video, "-vn", "-ac", "1", "-ar", str(SR),
         "-c:a", "pcm_s16le", wav],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def normaliser(mot):
    m = mot.lower().replace("\u2019", "'")
    return re.sub(r"[^\w'\-]", "", m)


def texte_normalise(texte):
    return " ".join(normaliser(x) for x in texte.split() if normaliser(x))


def liberer_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def appeler_avec_token(fn, token, **kw):
    """Compatibilite : selon la version, le parametre s'appelle token ou use_auth_token."""
    try:
        return fn(token=token, **kw)
    except TypeError:
        return fn(use_auth_token=token, **kw)


def classe_diarisation():
    """Selon la version de WhisperX, 'whisperx.diarize' n'est pas charge par
    'import whisperx' : on l'importe explicitement."""
    try:
        from whisperx.diarize import DiarizationPipeline
        return DiarizationPipeline
    except ImportError:
        return whisperx.DiarizationPipeline


def fin_de_phrase(mot):
    """Vrai si le mot se termine par un signe de fin de phrase."""
    return mot.rstrip("\"'\u2019\u201d\u00bb) ").endswith((".", "?", "!", "\u2026"))


def decouper_phrases(mots):
    phrases, cur = [], []
    for i, m in enumerate(mots):
        cur.append(i)
        if fin_de_phrase(m["word"]):
            phrases.append(cur)
            cur = []
    if cur:
        phrases.append(cur)
    return phrases


def locuteur_majoritaire(mots, ph, fillers):
    duree = {}
    for i in ph:
        if normaliser(mots[i]["word"]) in fillers:
            continue
        d = max(mots[i]["end"] - mots[i]["start"], 0.01)
        duree[mots[i]["speaker"]] = duree.get(mots[i]["speaker"], 0) + d
    return max(duree, key=duree.get) if duree else None


# --------------------------------------------------------------------------
# Empreintes vocales
# --------------------------------------------------------------------------
def charger_modele_empreinte(device, token):
    try:
        from pyannote.audio import Model
        modele = appeler_avec_token(
            Model.from_pretrained, token,
            checkpoint="pyannote/wespeaker-voxceleb-resnet34-LM")
        modele.eval().to(torch.device(device))
        return modele
    except Exception as e:  # noqa: BLE001
        info(f"[!] Modele d'empreinte indisponible ({e}). "
                   "Seules les regles du tour de parole seront utilisees.")
        return None


@torch.no_grad()
def empreinte(modele, audio, debut, fin, device, duree_min=0.5, duree_max=10.0):
    """Empreinte normalisee d'un passage. Les passages tres courts sont
    elargis symetriquement jusqu'a duree_min."""
    milieu = (debut + fin) / 2
    duree = min(max(fin - debut, duree_min), duree_max)
    i0 = int(max(0, (milieu - duree / 2)) * SR)
    i1 = min(len(audio), i0 + int(duree * SR))
    morceau = audio[i0:i1]
    if len(morceau) < int(0.2 * SR):
        return None
    x = torch.from_numpy(morceau.astype(np.float32))[None, None, :]
    e = modele(x.to(torch.device(device))).detach().cpu().numpy().reshape(-1)
    n = np.linalg.norm(e)
    return e / n if n > 0 else None


@torch.no_grad()
def empreinte_signal(modele, signal, device):
    """Empreinte d'un signal deja decoupe (au moins 0,5 s)."""
    if signal is None or len(signal) < int(0.5 * SR):
        return None
    x = torch.from_numpy(signal.astype(np.float32))[None, None, :]
    e = modele(x.to(torch.device(device))).detach().cpu().numpy().reshape(-1)
    n = np.linalg.norm(e)
    return e / n if n > 0 else None


def signal_mots(audio, mots, idx, duree_max=10.0):
    """Audio des seuls mots de la phrase, mis bout a bout : les pauses, bruits
    et voix de l'autre locuteur entre les mots ne polluent pas l'empreinte."""
    morceaux, total = [], 0.0
    duree_audio = len(audio) / SR
    for i in idx:
        a = int(max(0.0, mots[i]["start"] - 0.03) * SR)
        b = int(min(duree_audio, mots[i]["end"] + 0.03) * SR)
        if b - a < int(0.05 * SR):
            continue
        morceaux.append(audio[a:b])
        total += (b - a) / SR
        if total >= duree_max:
            break
    return np.concatenate(morceaux) if morceaux else None


def empreintes_reference(modele, audio, diar_df, device, duree_min_tour=3.0,
                         max_tours=8):
    """Empreinte moyenne par locuteur, calculee sur ses tours longs qui ne
    chevauchent aucun autre locuteur."""
    refs = {}
    tours = diar_df[["start", "end", "speaker"]].values.tolist()
    for loc in sorted({t[2] for t in tours}):
        propres = []
        for (d, f, s) in tours:
            if s != loc or f - d < duree_min_tour:
                continue
            chevauche = any(s2 != loc and not (f2 <= d or d2 >= f)
                            for (d2, f2, s2) in tours)
            if not chevauche:
                propres.append((f - d, d, f))
        propres.sort(reverse=True)
        embs = []
        for (_, d, f) in propres[:max_tours]:
            e = empreinte(modele, audio, d, f, device, duree_max=10.0)
            if e is not None:
                embs.append(e)
        if embs:
            moy = np.mean(embs, axis=0)
            refs[loc] = moy / np.linalg.norm(moy)
        else:
            info(f"[!] Aucun tour propre pour {loc} : pas d'empreinte de reference.")
    return refs


# --------------------------------------------------------------------------
# Mots et attribution des locuteurs
# --------------------------------------------------------------------------
def extraire_mots(resultat):
    """Liste de mots horodates SANS perdre de texte :
    - mot sans horodatage (chiffres, symboles) : horodatage interpole ;
    - segment que l'alignement n'a pas pu decouper en mots : le segment entier
      est garde, comme un seul bloc de texte avec les bornes du segment."""
    mots, nb_interp, nb_seg = [], 0, 0
    for seg in resultat["segments"]:
        ws = seg.get("words") or []
        if not ws:
            txt = (seg.get("text") or "").strip()
            if txt and "start" in seg and "end" in seg:
                mots.append({"word": txt, "start": seg["start"], "end": seg["end"]})
                nb_seg += 1
            continue
        for j, w in enumerate(ws):
            w = dict(w)
            if "start" not in w or "end" not in w:
                avant = next((x["end"] for x in reversed(ws[:j]) if "end" in x),
                             seg.get("start", 0.0))
                apres = next((x["start"] for x in ws[j + 1:] if "start" in x),
                             seg.get("end", avant))
                w["start"], w["end"] = avant, max(apres, avant)
                nb_interp += 1
            w.pop("speaker", None)      # le locuteur est recalcule plus bas
            mots.append(w)
    mots.sort(key=lambda w: w["start"])
    info(f"Mots sans horodatage interpoles : {nb_interp} | "
               f"segments non alignes gardes en bloc : {nb_seg}")
    return mots


def attribuer_locuteurs(mots, diar_df):
    """Chaque mot recoit le locuteur dont les tours de parole le recouvrent le
    plus ; sans recouvrement, le locuteur du tour le plus proche dans le temps.
    Un mot a donc TOUJOURS un locuteur (jamais 'INCONNU')."""
    if not mots:
        return
    tours = diar_df[["start", "end", "speaker"]].values.tolist()
    locuteurs = sorted({t[2] for t in tours})
    ws = np.array([m["start"] for m in mots], dtype=float)
    we = np.maximum(np.array([m["end"] for m in mots], dtype=float), ws + 0.01)
    n = len(mots)
    recouv = np.zeros((n, len(locuteurs)))
    dist = np.full((n, len(locuteurs)), np.inf)
    for j, loc in enumerate(locuteurs):
        ts = np.array([t[0] for t in tours if t[2] == loc], dtype=float)
        te = np.array([t[1] for t in tours if t[2] == loc], dtype=float)
        for a in range(0, n, 2000):
            sl = slice(a, a + 2000)
            o = np.minimum(we[sl, None], te[None, :]) - np.maximum(ws[sl, None], ts[None, :])
            recouv[sl, j] = np.clip(o, 0, None).sum(1)
            ecart = np.maximum(np.maximum(ts[None, :] - we[sl, None],
                                          ws[sl, None] - te[None, :]), 0)
            dist[sl, j] = ecart.min(1)
    for i, m in enumerate(mots):
        j = int(recouv[i].argmax()) if recouv[i].max() > 0 else int(dist[i].argmin())
        m["speaker"] = locuteurs[j]


def reparer_locuteurs(mots, valides):
    """Filet de securite : un mot sans locuteur valide recoit celui du mot
    valide le plus proche dans le temps. Renvoie le nombre de mots repares."""
    ok = [i for i, m in enumerate(mots) if m.get("speaker") in valides]
    if not ok:
        for m in mots:
            m["speaker"] = sorted(valides)[0]
        return len(mots)
    centres = np.array([(mots[i]["start"] + mots[i]["end"]) / 2 for i in ok])
    n = 0
    for m in mots:
        if m.get("speaker") in valides:
            continue
        c = (m["start"] + m["end"]) / 2
        m["speaker"] = mots[ok[int(np.abs(centres - c).argmin())]]["speaker"]
        m["correction"] = (m.get("correction", "") + " locuteur_voisin").strip()
        n += 1
    return n


# --------------------------------------------------------------------------
# Corrections des locuteurs
# --------------------------------------------------------------------------
def lisser_phrases(mots, fillers, max_mots=3, max_part=0.35, max_ecart=0.5):
    """Une phrase (delimitee par la ponctuation de Whisper) est en general dite
    par UNE seule personne. Si quelques mots (<= max_mots, <= max_part de la
    duree de la phrase) portent un autre locuteur, ils sont rattaches au
    locuteur majoritaire, sauf s'ils sont separes du reste par une vraie pause
    (> max_ecart) : c'est alors une prise de parole de l'autre."""
    nb = 0
    for ph in decouper_phrases(mots):
        idx = [i for i in ph if normaliser(mots[i]["word"]) not in fillers]
        if len(idx) < 2:
            continue
        duree = {}
        for i in idx:
            d = max(mots[i]["end"] - mots[i]["start"], 0.01)
            duree[mots[i]["speaker"]] = duree.get(mots[i]["speaker"], 0) + d
        if len(duree) < 2:
            continue
        maj = max(duree, key=duree.get)
        part_min = 1 - duree[maj] / sum(duree.values())
        mino = [i for i in idx if mots[i]["speaker"] != maj]
        majs = [j for j in idx if mots[j]["speaker"] == maj]
        if len(mino) <= max_mots and (part_min <= max_part
                                      or (len(mino) == 1 and len(majs) >= 3)):
            for i in mino:
                ecart = min(max(mots[j]["start"] - mots[i]["end"],
                                mots[i]["start"] - mots[j]["end"], 0.0)
                            for j in majs)
                if ecart > max_ecart:
                    continue
                mots[i]["correction"] = f"phrase:{mots[i]['speaker']}->{maj}"
                mots[i]["speaker"] = maj
                nb += 1
    info(f"Mots rattaches a la bonne phrase (lissage) : {nb}")


def verifier_phrases(mots, audio, modele, device, refs, fillers, duree_max=8.0,
                     marge=0.10, min_mots=2, taille_max=2):
    """La diarisation rate souvent une phrase (ou deux) de l'un enclavee dans le
    long tour de l'autre. Pour chaque bloc de 1 a 2 phrases (<= duree_max) dont
    la phrase precedente ET la suivante ont le meme locuteur, on compare sa voix
    aux deux empreintes de reference. L'empreinte est calculee sur l'audio des
    seuls mots. Si l'autre locuteur est nettement plus proche (ecart >= marge),
    le bloc lui est attribue et marque pour relecture isolee."""
    phrases = decouper_phrases(mots)
    maj = [locuteur_majoritaire(mots, ph, fillers) for ph in phrases]
    nb, k = 0, 1
    barre = barre_interne(total=max(len(phrases) - 2, 0), desc="Verification des phrases",
                 leave=False)
    while k < len(phrases) - 1:
        reassigne = False
        cur = maj[k]
        for L in range(1, taille_max + 1):
            fin = k + L - 1
            if fin >= len(phrases) - 1 or cur is None:
                break
            if any(maj[j] != cur for j in range(k, fin + 1)):
                break
            if maj[k - 1] != cur or maj[fin + 1] != cur:
                continue
            idx = [i for j in range(k, fin + 1) for i in phrases[j]]
            utiles = [i for i in idx if normaliser(mots[i]["word"]) not in fillers]
            if len(utiles) < min_mots:
                continue
            d, f = mots[idx[0]]["start"], mots[idx[-1]]["end"]
            if f - d > duree_max:
                continue
            e = empreinte_signal(modele, signal_mots(audio, mots, utiles), device)
            if e is None or cur not in refs:
                continue
            scores = {loc: float(np.dot(e, r)) for loc, r in refs.items()}
            meilleur = max(scores, key=scores.get)
            if meilleur != cur and scores[meilleur] - scores[cur] >= marge:
                for i in idx:
                    mots[i]["correction"] = f"phrase_courte:{cur}->{meilleur}"
                    mots[i]["speaker"] = meilleur
                    mots[i]["relire"] = True
                for j in range(k, fin + 1):
                    maj[j] = meilleur
                nb += 1
                info(f"  [{d:.1f}-{f:.1f}s] {cur} -> {meilleur} "
                           f"(ecart {scores[meilleur] - scores[cur]:.2f}) : "
                           + " ".join(mots[i]["word"] for i in idx))
                barre.update(fin + 1 - k)
                k = fin + 1
                reassigne = True
                break
        if not reassigne:
            barre.update(1)
            k += 1
    barre.close()
    info(f"Phrases courtes reattribuees : {nb}")


def trouver_runs(mots, fillers, ecart_max=0.6, taille_max=4):
    """Suites consecutives de mots-interjections."""
    runs, courant = [], []
    for i, m in enumerate(mots):
        if normaliser(m["word"]) in fillers:
            if courant and m["start"] - mots[courant[-1]]["end"] > ecart_max:
                runs.append(courant)
                courant = []
            courant.append(i)
        elif courant:
            runs.append(courant)
            courant = []
    if courant:
        runs.append(courant)
    return [r for r in runs if len(r) <= taille_max]


def regle_tour_de_parole(mots, run, loc_actuel, fillers, fenetre=2.0):
    """Vrai si l'interjection est enfermee dans le tour de parole de son
    locuteur actuel (donc probablement dite par l'autre)."""
    d, f = mots[run[0]]["start"], mots[run[-1]]["end"]
    idx_set = set(run)
    avant = apres = False
    for j in range(run[0] - 1, -1, -1):
        if j in idx_set or normaliser(mots[j]["word"]) in fillers:
            continue
        avant = mots[j]["speaker"] == loc_actuel and d - mots[j]["end"] <= fenetre
        break
    for j in range(run[-1] + 1, len(mots)):
        if j in idx_set or normaliser(mots[j]["word"]) in fillers:
            continue
        apres = mots[j]["speaker"] == loc_actuel and mots[j]["start"] - f <= fenetre
        break
    return avant and apres


def traiter_interjections(mots, audio, modele, device, refs, fillers,
                          marge_min, ecart_isole=0.25):
    locuteurs = sorted({m["speaker"] for m in mots})
    nb_corrigees = 0
    for k, run in enumerate(barre_interne(trouver_runs(mots, fillers),
                                 desc="Interjections", leave=False)):
        d, f = mots[run[0]]["start"], mots[run[-1]]["end"]
        actuel = mots[run[0]]["speaker"]
        nouveau, methode = actuel, ""

        if modele is not None and len(refs) >= 2:
            e = empreinte(modele, audio, d, f, device)
            if e is not None:
                scores = {loc: float(np.dot(e, r)) for loc, r in refs.items()}
                tri = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                if tri[0][1] - tri[1][1] >= marge_min:
                    nouveau, methode = tri[0][0], "empreinte"

        if not methode and len(locuteurs) == 2:
            if regle_tour_de_parole(mots, run, actuel, fillers):
                nouveau = [l for l in locuteurs if l != actuel][0]
                methode = "regle"

        change = nouveau != actuel
        prec = mots[run[0] - 1]["end"] if run[0] > 0 else None
        suiv = mots[run[-1] + 1]["start"] if run[-1] + 1 < len(mots) else None
        isole = ((prec is None or d - prec > ecart_isole)
                 or (suiv is None or suiv - f > ecart_isole))
        if change or isole:
            for i in run:
                mots[i]["speaker"] = nouveau
                mots[i]["interj"] = k
                mots[i]["correction"] = (f"{methode}:{actuel}->{nouveau}"
                                         if change else "")
            nb_corrigees += int(change)
    info(f"Interjections reattribuees : {nb_corrigees}")


# --------------------------------------------------------------------------
# Relecture : retranscription isolee
# --------------------------------------------------------------------------
def charger_relecteur(nom_modele, device, compute):
    from faster_whisper import WhisperModel
    return WhisperModel(nom_modele, device=device, compute_type=compute)


def lire_isole(fw, audio, d, f, langue, prompt, marge=0.15):
    """Retranscrit [d, f] SEUL, sans detection de voix ni contexte. Renvoie des
    mots horodates (temps absolus), ou [] si rien de fiable."""
    d0, f0 = max(0.0, d - marge), min(len(audio) / SR, f + marge)
    morceau = audio[int(d0 * SR):int(f0 * SR)]
    if len(morceau) < int(0.3 * SR):
        return []
    try:
        segs, _ = fw.transcribe(morceau, language=langue, initial_prompt=prompt,
                                beam_size=5, vad_filter=False, word_timestamps=True,
                                condition_on_previous_text=False)
        segs = list(segs)
    except Exception as e:  # noqa: BLE001
        info(f"[!] Relecture impossible ({e})")
        return []
    sortie = []
    for sg in segs:
        t = sg.text.strip().lower()
        if (not t or getattr(sg, "no_speech_prob", 0.0) > 0.7
                or getattr(sg, "avg_logprob", 0.0) < -1.2
                or any(h in t for h in HALLUCINATIONS)):
            continue
        for w in (sg.words or []):
            mot = w.word.strip()
            if mot:
                sortie.append({"word": mot, "start": d0 + w.start, "end": d0 + w.end})
    return sortie


def blocs_a_relire(mots):
    blocs, a = [], None
    for i, m in enumerate(mots + [{}]):
        if m.get("relire"):
            if a is None:
                a = i
        elif a is not None:
            blocs.append((a, i - 1))
            a = None
    return blocs


def relire_phrases(mots, fw, audio, langue, prompt, sim_min=0.35):
    """Retranscrit isolement les phrases reattribuees a l'autre locuteur. Le
    nouveau texte est garde seulement s'il ressemble a l'ancien (sinon on a
    probablement lu la voix de l'autre locuteur)."""
    nb = 0
    for (a, b) in barre_interne(list(reversed(blocs_a_relire(mots))), desc="Relecture isolee",
                       leave=False):
        anciens = mots[a:b + 1]
        d, f, loc = anciens[0]["start"], anciens[-1]["end"], anciens[0]["speaker"]
        ancien = " ".join(m["word"] for m in anciens)
        nouveaux = lire_isole(fw, audio, d, f, langue, prompt)
        if not nouveaux:
            continue
        texte = " ".join(w["word"] for w in nouveaux)
        sim = difflib.SequenceMatcher(None, texte_normalise(ancien),
                                      texte_normalise(texte)).ratio()
        if sim < sim_min or texte_normalise(texte) == texte_normalise(ancien):
            continue
        mots[a:b + 1] = [{"word": w["word"], "start": w["start"], "end": w["end"],
                          "speaker": loc,
                          "correction": f"relu_isole (avant : {ancien})"}
                         for w in nouveaux]
        nb += 1
        info(f"  relu [{d:.1f}-{f:.1f}s] : {ancien}  =>  {texte}")
    info(f"Phrases relues isolement et modifiees : {nb}")


def tours_non_couverts(mots, diar_df, duree_min=0.6, couverture_max=0.25,
                       ecart_fusion=0.4, duree_morceau_min=0.5):
    """Parties des tours de parole (PAR LOCUTEUR) ou ce locuteur n'a pas de
    texte : voix ratee par la detection de voix, ou parole chevauchee par
    l'autre locuteur (dont les mots 'couvraient' le passage)."""
    par_loc = {}
    for m in mots:
        par_loc.setdefault(m["speaker"], []).append((m["start"], m["end"]))
    debuts = {loc: [a for a, _ in v] for loc, v in par_loc.items()}

    morceaux = []
    for d, f, loc in diar_df[["start", "end", "speaker"]].itertuples(index=False):
        if f - d < duree_min:
            continue
        iv = par_loc.get(loc, [])
        i0 = bisect.bisect_left(debuts.get(loc, []), d - 30.0)
        couvert = []
        for a, b in iv[i0:]:
            if a >= f:
                break
            if b <= d:
                continue
            if couvert and a <= couvert[-1][1]:
                couvert[-1][1] = max(couvert[-1][1], b)
            else:
                couvert.append([a, b])
        deja = sum(max(0.0, min(f, b) - max(d, a)) for a, b in couvert)
        if deja / (f - d) >= couverture_max:
            continue
        pos = d
        for a, b in couvert:
            if a > pos:
                morceaux.append([pos, a, loc])
            pos = max(pos, b)
        if pos < f:
            morceaux.append([pos, f, loc])

    morceaux.sort(key=lambda x: (x[2], x[0]))
    fus = []
    for d, f, loc in morceaux:
        if fus and fus[-1][2] == loc and d - fus[-1][1] <= ecart_fusion:
            fus[-1][1] = max(fus[-1][1], f)
        else:
            fus.append([d, f, loc])
    fus = [(d, f, loc) for d, f, loc in fus if f - d >= duree_morceau_min]
    return sorted(fus)


def recuperer_tours(mots, morceaux, fw, audio, langue, prompt):
    """Retranscrit les voix manquantes. Les mots deja presents autour du passage
    (de l'autre locuteur, quand il y a chevauchement) ne sont pas dupliques :
    on ne garde que ce qui est NOUVEAU."""
    nouveaux = []
    for (d, f, loc) in barre_interne(morceaux, desc="Paroles manquantes", leave=False):
        lus = [w for w in lire_isole(fw, audio, d, f, langue, prompt)
               if d - 0.05 <= (w["start"] + w["end"]) / 2 <= f + 0.05]
        deja = {normaliser(m["word"]) for m in mots
                if m["end"] > d - 0.5 and m["start"] < f + 0.5}
        restants = [w for w in lus if normaliser(w["word"]) not in deja]
        if not restants:
            continue
        for w in restants:
            nouveaux.append({"word": w["word"], "start": w["start"], "end": w["end"],
                             "speaker": loc, "correction": "rattrape"})
        info(f"  [{d:.1f}-{f:.1f}s] {loc} retrouve : "
                   + " ".join(w["word"] for w in restants)[:100])
    mots.extend(nouveaux)
    mots.sort(key=lambda w: w["start"])
    info(f"Mots retrouves : {len(nouveaux)}")


# --------------------------------------------------------------------------
# Dictionnaire de corrections, regroupement, controle
# --------------------------------------------------------------------------
def lire_corrections(chemin):
    """Fichier texte : une regle par ligne, 'avant = apres' (# = commentaire)."""
    regles = []
    if not chemin:
        return regles
    with open(os.path.expanduser(chemin), encoding="utf-8") as fh:
        for ligne in fh:
            ligne = ligne.strip()
            if not ligne or ligne.startswith("#") or "=" not in ligne:
                continue
            a, b = ligne.split("=", 1)
            regles.append((re.compile(r"\b" + re.escape(a.strip()) + r"\b",
                                      re.IGNORECASE), b.strip()))
    return regles


def appliquer_corrections(texte, regles):
    n = 0
    for rx, rep in regles:
        texte, k = rx.subn(rep, texte)
        n += k
    return texte, n


def regrouper(mots, seuil_pause, par_phrase=True):
    segs = []
    for m in mots:
        loc, cle = m["speaker"], m.get("interj")
        s = segs[-1] if segs else None
        if (s is None or s["locuteur"] != loc or s["interj"] != cle
                or m["start"] - s["fin"] > seuil_pause
                or (par_phrase and s["phrase_close"])):
            s = {"debut": m["start"], "fin": m["end"], "locuteur": loc,
                 "mots": [], "interj": cle, "corrections": [], "scores": [],
                 "phrase_close": False}
            segs.append(s)
        s["fin"] = max(s["fin"], m["end"])
        s["mots"].append(m["word"])
        c = m.get("correction")
        if c and c not in s["corrections"]:
            s["corrections"].append(c)
        if m.get("score") is not None:
            s["scores"].append(float(m["score"]))
        s["phrase_close"] = fin_de_phrase(m["word"])
    segs.sort(key=lambda s: s["debut"])
    return segs


def signaler(segments, seuil_score=0.4, chevauchement_min=0.4):
    """Raisons pour lesquelles une ligne merite d'etre verifiee a la main."""
    raisons = [set() for _ in segments]
    for i, s in enumerate(segments):
        if s["corrections"]:
            raisons[i].add("corrigee automatiquement")
        if s["score"] is not None and s["score"] < seuil_score:
            raisons[i].add("faible confiance de l'alignement")
        j = i + 1
        while j < len(segments) and segments[j]["debut"] < s["fin"]:
            t = segments[j]
            if (t["locuteur"] != s["locuteur"]
                    and min(s["fin"], t["fin"]) - t["debut"] >= chevauchement_min):
                raisons[i].add("chevauche l'autre voix")
                raisons[j].add("chevauche l'autre voix")
            j += 1
    return raisons


# --------------------------------------------------------------------------
# Profils des passes : chaque passe utilise des reglages DIFFERENTS
#   audio    : "brut" ou "norm" (normalise en volume par ffmpeg/dynaudnorm)
#   vad_*    : sensibilite de la detection de voix (bas = plus sensible)
#   beam     : finesse du decodage de Whisper
#   modele   : "principal" (--modele) ou "second" (--modele2, sinon le principal)
#   seed     : graine aleatoire (diarisation)
# La tentative k utilise les profils 2k-1 et 2k de cette liste.
# --------------------------------------------------------------------------
PROFILS = [
    {"nom": "A1", "audio": "brut", "vad_onset": 0.25, "vad_offset": 0.20, "beam": 5, "modele": "principal", "seed": 0},
    {"nom": "B1", "audio": "norm", "vad_onset": 0.35, "vad_offset": 0.25, "beam": 5, "modele": "second",    "seed": 1},
    {"nom": "A2", "audio": "brut", "vad_onset": 0.30, "vad_offset": 0.22, "beam": 4, "modele": "principal", "seed": 2},
    {"nom": "B2", "audio": "norm", "vad_onset": 0.20, "vad_offset": 0.15, "beam": 5, "modele": "second",    "seed": 3},
    {"nom": "A3", "audio": "norm", "vad_onset": 0.25, "vad_offset": 0.20, "beam": 3, "modele": "principal", "seed": 4},
    {"nom": "B3", "audio": "brut", "vad_onset": 0.22, "vad_offset": 0.18, "beam": 5, "modele": "second",    "seed": 5},
]

ENTETE_CSV = ["debut_s", "fin_s", "locuteur", "texte", "type", "pause_avant_s",
              "correction", "score_moyen"]


def extraire_audio_filtre(video, wav, filtre=None):
    cmd = ["ffmpeg", "-y", "-i", video, "-vn", "-ac", "1", "-ar", str(SR)]
    if filtre:
        cmd += ["-af", filtre]
    subprocess.run(cmd + ["-c:a", "pcm_s16le", wav], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# --------------------------------------------------------------------------
# Une passe complete : diarisation -> transcription -> corrections
# --------------------------------------------------------------------------
def traiter(audio, nom_modele, profil, a, ctx, emb_modele, etapes=None):
    """Une passe complete avec les reglages du profil. Renvoie la liste des mots."""
    device, compute = ctx["device"], ctx["compute"]
    torch.manual_seed(profil["seed"])
    np.random.seed(profil["seed"])

    # diarisation
    _etape(etapes, profil, "Diarisation")
    info(f"  [{profil['nom']}] diarisation...")
    diar = appeler_avec_token(classe_diarisation(), ctx["token"], device=device)
    diar_df = diar(audio, min_speakers=a.nb_locuteurs, max_speakers=a.nb_locuteurs)
    del diar
    liberer_gpu()
    valides = set(diar_df["speaker"])
    _fin_etape(etapes)

    # transcription
    _etape(etapes, profil, "Transcription (etape longue)")
    info(f"  [{profil['nom']}] transcription ({nom_modele}, "
               f"audio {profil['audio']}, beam {profil['beam']})...")
    modele = whisperx.load_model(
        nom_modele, device, compute_type=compute, language=a.langue,
        asr_options={"initial_prompt": ctx["prompt"], "suppress_numerals": False,
                     "beam_size": profil["beam"]},
        vad_options={"vad_onset": profil["vad_onset"],
                     "vad_offset": profil["vad_offset"]},
    )
    try:
        resultat = modele.transcribe(audio, batch_size=a.batch, language=a.langue,
                                     print_progress=VERBEUX)
    except TypeError:
        resultat = modele.transcribe(audio, batch_size=a.batch, language=a.langue)
    del modele
    liberer_gpu()
    _fin_etape(etapes)

    # alignement + attribution des locuteurs
    _etape(etapes, profil, "Alignement")
    m_align, meta = whisperx.load_align_model(language_code=a.langue, device=device)
    resultat = whisperx.align(resultat["segments"], m_align, meta, audio, device,
                              return_char_alignments=False)
    del m_align
    liberer_gpu()
    mots = extraire_mots(resultat)
    attribuer_locuteurs(mots, diar_df)
    _fin_etape(etapes)

    # corrections des locuteurs
    _etape(etapes, profil, "Corrections des locuteurs")
    refs = {}
    if emb_modele is not None:
        refs = empreintes_reference(emb_modele, audio, diar_df, device)
    if not a.sans_lissage:
        lisser_phrases(mots, ctx["fillers"], max_mots=a.max_mots_lissage)
    if not a.sans_verif_phrases and emb_modele is not None and len(refs) >= 2:
        verifier_phrases(mots, audio, emb_modele, device, refs, ctx["fillers"],
                         duree_max=a.duree_phrase_courte, marge=a.marge_phrase)
    traiter_interjections(mots, audio, emb_modele, device, refs, ctx["fillers"],
                          a.marge_empreinte)
    _fin_etape(etapes)

    # relecture isolee + voix manquantes
    if not a.sans_relecture:
        _etape(etapes, profil, "Relecture isolee")
        fw = charger_relecteur(nom_modele, device, compute)
        relire_phrases(mots, fw, audio, a.langue, ctx["prompt_relecture"])
        morceaux = tours_non_couverts(mots, diar_df, duree_min=a.duree_min_tour)
        recuperer_tours(mots, morceaux, fw, audio, a.langue, ctx["prompt_relecture"])
        del fw
        liberer_gpu()
        _fin_etape(etapes)

    reparer_locuteurs(mots, valides)
    return mots


# --------------------------------------------------------------------------
# Comparaison de deux passes
# --------------------------------------------------------------------------
def jetons(mots):
    """(mot normalise, indice du mot, debut) pour chaque mot de la transcription."""
    res = []
    for i, m in enumerate(mots):
        for w in str(m["word"]).split():
            t = normaliser(w)
            if t:
                res.append((t, i, m["start"]))
    return res


def apparier(mots_a, mots_b, fenetre=30.0):
    """Apparie les mots identiques des deux transcriptions, fenetre de temps par
    fenetre (meme audio -> memes instants). Renvoie (paires, comptes) avec
    paires = [(i, j, fenetre)] et comptes = {fenetre: [nb_mots_A, nb_mots_B]}."""
    ja, jb = jetons(mots_a), jetons(mots_b)

    def par_fenetre(j):
        d = {}
        for k, (_, _, s) in enumerate(j):
            d.setdefault(int(s // fenetre), []).append(k)
        return d

    wa, wb = par_fenetre(ja), par_fenetre(jb)
    comptes = {f: [len(wa.get(f, [])), len(wb.get(f, []))] for f in set(wa) | set(wb)}
    paires = []
    for f in sorted(set(wa) & set(wb)):
        ka, kb = wa[f], wb[f]
        sm = difflib.SequenceMatcher(None, [ja[k][0] for k in ka],
                                     [jb[k][0] for k in kb], autojunk=False)
        for a0, b0, n in sm.get_matching_blocks():
            for d in range(n):
                paires.append((ja[ka[a0 + d]][1], jb[kb[b0 + d]][1], f))
    return paires, comptes


def meilleure_correspondance(mots_a, mots_b, paires):
    """Les etiquettes SPEAKER_xx sont arbitraires d'une diarisation a l'autre :
    on cherche la correspondance B -> A qui maximise l'accord."""
    la = sorted({m["speaker"] for m in mots_a})
    lb = sorted({m["speaker"] for m in mots_b})
    conf = {(x, y): 0 for x in la for y in lb}
    for i, j, _ in paires:
        conf[(mots_a[i]["speaker"], mots_b[j]["speaker"])] += 1
    if len(lb) <= len(la) <= 6:
        meilleur, score = None, -1
        for perm in itertools.permutations(la, len(lb)):
            sc = sum(conf[(perm[k], lb[k])] for k in range(len(lb)))
            if sc > score:
                score, meilleur = sc, dict(zip(lb, perm))
        return meilleur
    return {y: max(la, key=lambda x: conf[(x, y)]) for y in lb}


def aligner_locuteurs(ref, mots):
    """Renomme les locuteurs de `mots` pour qu'ils correspondent a ceux de `ref`."""
    if ref is mots:
        return
    paires, _ = apparier(ref, mots)
    corr = meilleure_correspondance(ref, mots, paires)
    for m in mots:
        m["speaker"] = corr.get(m["speaker"], m["speaker"])


def concordance(mots_a, mots_b):
    """texte   : mots identiques / mots (Dice)
       locuteur: parmi les mots identiques, part attribuee au meme locuteur
       global  : mots identiques ET meme locuteur / mots   <- critere d'arret"""
    paires, comptes = apparier(mots_a, mots_b)
    na = sum(c[0] for c in comptes.values())
    nb = sum(c[1] for c in comptes.values())
    vide = {"texte": 0.0, "locuteur": 0.0, "global": 0.0, "paires": [],
            "accords": [], "comptes": comptes}
    if not paires or na + nb == 0:
        return vide
    corr = meilleure_correspondance(mots_a, mots_b, paires)
    accords = [corr.get(mots_b[j]["speaker"]) == mots_a[i]["speaker"]
               for i, j, _ in paires]
    return {"texte": 2 * len(paires) / (na + nb),
            "locuteur": sum(accords) / len(paires),
            "global": 2 * sum(accords) / (na + nb),
            "paires": paires, "accords": accords, "comptes": comptes}


def score_moyen(mots):
    s = [float(m["score"]) for m in mots if m.get("score") is not None]
    return float(np.mean(s)) if s else 0.0


def resume_fenetre(mots, f, noms, fenetre=30.0):
    """Texte d'une fenetre de temps, avec le nom du locuteur a chaque changement."""
    out, dernier = [], None
    for m in mots:
        if int(m["start"] // fenetre) != f:
            continue
        if m["speaker"] != dernier:
            out.append(f"[{noms.get(m['speaker'], m['speaker'])}]")
            dernier = m["speaker"]
        out.append(str(m["word"]))
    return " ".join(out)


def ecrire_divergences(mots_a, mots_b, c, noms, chemin, seuil=0.85, fenetre=30.0):
    """Fenetres de 30 s ou les deux passes different : a relire en priorite."""
    par_f = {}
    for (i, j, f), ok in zip(c["paires"], c["accords"]):
        par_f[f] = par_f.get(f, 0) + int(ok)
    lignes = []
    for f, (na, nb) in sorted(c["comptes"].items()):
        if na + nb < 10:
            continue
        conc = 2 * par_f.get(f, 0) / (na + nb)
        if conc < seuil:
            lignes.append([round(f * fenetre, 1), round((f + 1) * fenetre, 1),
                           round(conc, 2), resume_fenetre(mots_a, f, noms, fenetre),
                           resume_fenetre(mots_b, f, noms, fenetre)])
    with open(chemin, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["debut_s", "fin_s", "concordance", "passe_A", "passe_B"])
        w.writerows(lignes)
    return len(lignes)


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------
def construire_segments(mots, seuil_pause, par_phrase, regles):
    segments = regrouper(mots, seuil_pause, par_phrase=par_phrase)
    for s in segments:
        texte, n = appliquer_corrections(" ".join(s["mots"]), regles)
        if n:
            s["corrections"].append("dictionnaire")
        s["texte"] = texte
        s["score"] = float(np.mean(s["scores"])) if s["scores"] else None
    return segments


def ecrire_csv(segments, chemin, noms, chemin_verif=None):
    raisons = signaler(segments)
    lignes, verif, fin_max = [], [], None
    for s, rs in zip(segments, raisons):
        pause = max(round(s["debut"] - fin_max, 2), 0) if fin_max is not None else 0
        ligne = [round(s["debut"], 2), round(s["fin"], 2),
                 noms.get(s["locuteur"], s["locuteur"]), s["texte"],
                 "interjection" if s["interj"] is not None else "parole", pause,
                 " | ".join(s["corrections"]),
                 "" if s["score"] is None else round(s["score"], 2)]
        lignes.append(ligne)
        if rs:
            verif.append(ligne + ["; ".join(sorted(rs))])
        fin_max = s["fin"] if fin_max is None else max(fin_max, s["fin"])
    with open(chemin, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(ENTETE_CSV)
        w.writerows(lignes)
    if chemin_verif:
        with open(chemin_verif, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(ENTETE_CSV + ["raison"])
            w.writerows(verif)
    return len(verif)


def sauver_mots(mots, chemin):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return str(o)
    with open(chemin, "w", encoding="utf-8") as fh:
        json.dump(mots, fh, ensure_ascii=False, default=conv)


def charger_mots(chemin):
    with open(chemin, encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# Evaluation contre la transcription de reference
# --------------------------------------------------------------------------
TEXT_HEADERS = {
    "texte", "text", "transcription", "transcript", "annotationvalue",
    "annotation", "content", "contenu", "paroles", "parole", "utterance",
    "segmenttext", "sentence", "phrase", "verbatim",
}
SPEAKER_HEADERS = {
    "locuteur", "speaker", "speakername", "interlocuteur", "participant",
    "tier", "tiername", "nomlocuteur", "role", "acteur",
}
START_HEADERS = {
    "debuts", "debut", "start", "starts", "starttime", "begin", "begintime",
    "temps", "time", "debutsec", "startsec", "beginsec", "debutsecondes",
}


def cle_entete(valeur):
    valeur = unicodedata.normalize("NFKD", str(valeur or ""))
    valeur = "".join(c for c in valeur if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", valeur.lower())


def jetons_texte(texte):
    """Tokenisation stable : ignore la ponctuation et la casse, conserve les accents."""
    texte = unicodedata.normalize("NFKC", str(texte or "")).lower()
    texte = texte.replace("\u2019", "'").replace("\u2018", "'").replace("\u2013", "-").replace("\u2014", "-")
    return re.findall(r"[^\W_]+", texte, flags=re.UNICODE)


def cle_locuteur(valeur):
    valeur = unicodedata.normalize("NFKD", str(valeur or "")).lower()
    valeur = "".join(c for c in valeur if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", valeur)


def temps_reference(valeur):
    """Convertit un temps numerique ou HH:MM:SS.mmm en secondes pour trier les lignes."""
    valeur = str(valeur or "").strip().replace(",", ".")
    if not valeur:
        return None
    try:
        return float(valeur)
    except ValueError:
        pass
    morceaux = valeur.split(":")
    if len(morceaux) in (2, 3):
        try:
            nums = [float(x) for x in morceaux]
            if len(nums) == 2:
                return nums[0] * 60 + nums[1]
            return nums[0] * 3600 + nums[1] * 60 + nums[2]
        except ValueError:
            return None
    return None


def trouver_colonne(entetes, candidats):
    par_cle = {cle_entete(h): h for h in entetes}
    for cle in candidats:
        if cle in par_cle:
            return par_cle[cle]
    return None


def charger_reference(chemin):
    """Charge une reference TXT, CSV ou TSV. Pour un tableau, detecte texte, temps et locuteur."""
    if not os.path.isfile(chemin):
        raise FileNotFoundError(f"Fichier de reference introuvable : {chemin}")
    with open(chemin, "r", encoding="utf-8-sig", errors="replace", newline="") as fh:
        contenu = fh.read()
    if not contenu.strip():
        raise ValueError("Le fichier de reference est vide.")

    extension = os.path.splitext(chemin)[1].lower()
    lignes = contenu.splitlines()
    echantillon = "\n".join(lignes[:20])
    dialecte = None
    try:
        dialecte = csv.Sniffer().sniff(echantillon, delimiters="\t,;|")
    except csv.Error:
        pass

    lignes_table = extension in {".csv", ".tsv", ".tab"}
    entetes_test = []
    if dialecte and lignes:
        try:
            entetes_test = next(csv.reader([lignes[0]], dialecte))
        except (csv.Error, StopIteration):
            entetes_test = []
    entetes_connues = any(cle_entete(h) in TEXT_HEADERS for h in entetes_test)
    # Les exports tabules d'ELAN portent parfois une extension .txt.
    lignes_table = lignes_table or entetes_connues

    jetons = []
    locuteurs = []
    a_locuteurs = False
    if lignes_table:
        if not dialecte:
            dialecte = csv.excel_tab if "\t" in echantillon else csv.excel
        lecteur = csv.DictReader(contenu.splitlines(), dialect=dialecte)
        if not lecteur.fieldnames:
            raise ValueError("Impossible de lire les entetes du fichier CSV/TSV de reference.")
        champ_texte = trouver_colonne(lecteur.fieldnames, TEXT_HEADERS)
        if champ_texte is None:
            # Fallback : choisir la colonne non-temporelle contenant le plus de texte.
            exclusions = START_HEADERS | SPEAKER_HEADERS | {
                "fin", "fins", "end", "endtime", "duration", "duree", "type", "id", "index",
            }
            options = [h for h in lecteur.fieldnames if cle_entete(h) not in exclusions]
            if len(options) == 1:
                champ_texte = options[0]
            else:
                scores = []
                for h in options:
                    vals = [str(row.get(h, "") or "").strip()
                            for row in csv.DictReader(contenu.splitlines(), dialect=dialecte)]
                    scores.append((sum(len(v) for v in vals), h))
                if scores:
                    champ_texte = max(scores)[1]
        if champ_texte is None:
            raise ValueError("Colonne de texte non detectee. Utilisez une colonne nommee texte ou transcription.")

        champ_locuteur = trouver_colonne(lecteur.fieldnames, SPEAKER_HEADERS)
        champ_debut = trouver_colonne(lecteur.fieldnames, START_HEADERS)
        enregistrements = list(lecteur)
        if champ_debut:
            temporises = [(temps_reference(r.get(champ_debut, "")), i, r)
                          for i, r in enumerate(enregistrements)]
            if all(t[0] is not None for t in temporises):
                enregistrements = [t[2] for t in sorted(temporises, key=lambda t: (t[0], t[1]))]
        for row in enregistrements:
            texte = str(row.get(champ_texte, "") or "").strip()
            ts = jetons_texte(texte)
            if not ts:
                continue
            jetons.extend(ts)
            label = str(row.get(champ_locuteur, "") or "").strip() if champ_locuteur else ""
            locuteurs.extend([cle_locuteur(label)] * len(ts))
        a_locuteurs = bool(champ_locuteur and any(locuteurs))
        if not a_locuteurs:
            locuteurs = []
    else:
        # Nettoyage elementaire pour TXT, SRT et VTT.
        textes = []
        for ligne in lignes:
            ligne = ligne.strip()
            if not ligne or ligne.upper() == "WEBVTT":
                continue
            if re.fullmatch(r"\d+", ligne):
                continue
            if "-->" in ligne:
                continue
            ligne = re.sub(r"<[^>]+>", " ", ligne)
            # Dans les fichiers texte avec des tours marques "Psy: ..." ou "Sujet: ...",
            # retire le nom de role en debut de ligne pour ne pas le compter comme une parole.
            ligne = re.sub(
                r"^\s*(?:SPEAKER_\d+|Psy(?:chologue)?|Sujet|Patient|Interviewer|Therapist|Interlocuteur\s*\d*)\s*:\s*",
                "", ligne, flags=re.IGNORECASE,
            )
            textes.append(ligne)
        jetons = jetons_texte(" ".join(textes))

    if not jetons:
        raise ValueError("Aucun mot n'a ete trouve dans le fichier de reference.")
    return {"tokens": jetons, "speakers": locuteurs if a_locuteurs else None,
            "has_speakers": a_locuteurs, "path": os.path.abspath(chemin)}


def distance_levenshtein(reference, hypothese):
    """Distance d'edition exacte; utilise RapidFuzz si disponible, sinon DP en memoire reduite."""
    try:
        from rapidfuzz.distance import Levenshtein
        return int(Levenshtein.distance(reference, hypothese))
    except ImportError:
        pass
    if len(reference) < len(hypothese):
        reference, hypothese = hypothese, reference
    precedent = list(range(len(hypothese) + 1))
    for i, mot_ref in enumerate(reference, 1):
        courant = [i]
        for j, mot_hyp in enumerate(hypothese, 1):
            courant.append(min(
                courant[-1] + 1,
                precedent[j] + 1,
                precedent[j - 1] + (mot_ref != mot_hyp),
            ))
        precedent = courant
    return precedent[-1]


def blocs_egaux(reference, hypothese):
    """Blocs de mots identiques alignes, pour calculer le score des locuteurs."""
    try:
        from rapidfuzz.distance import Levenshtein
        ops = Levenshtein.opcodes(reference, hypothese)
        return [(i1, i2, j1, j2) for tag, i1, i2, j1, j2 in ops if tag == "equal"]
    except (ImportError, AttributeError, TypeError):
        sm = difflib.SequenceMatcher(None, reference, hypothese, autojunk=False)
        return [(i, i + n, j, j + n) for tag, i, j, n in sm.get_opcodes() if tag == "equal"]


def mots_et_locuteurs_candidat(mots, noms):
    tokens, labels = [], []
    for mot in mots:
        t = jetons_texte(mot.get("word", ""))
        loc = noms.get(mot.get("speaker", ""), mot.get("speaker", ""))
        tokens.extend(t)
        labels.extend([cle_locuteur(loc)] * len(t))
    return tokens, labels


def evaluer_candidat(mots, reference, noms, poids_locuteur=0.20):
    """Calcule WER et score texte; ajoute un score locuteur si la reference est exploitable."""
    tokens_hyp, labels_hyp = mots_et_locuteurs_candidat(mots, noms)
    tokens_ref = reference["tokens"]
    if not tokens_ref:
        raise ValueError("La reference ne contient aucun mot.")
    distance = distance_levenshtein(tokens_ref, tokens_hyp)
    wer = distance / len(tokens_ref)
    score_texte = max(0.0, 100.0 * (1.0 - wer))

    score_locuteur = None
    if reference.get("has_speakers") and reference.get("speakers"):
        ref_labels = {x for x in reference["speakers"] if x}
        hyp_labels = {x for x in labels_hyp if x}
        # Les noms des locuteurs doivent etre comparables; sinon on ne calcule pas de faux score.
        if ref_labels and ref_labels.issubset(hyp_labels):
            egaux = blocs_egaux(tokens_ref, tokens_hyp)
            corrects = total = 0
            labels_ref = reference["speakers"]
            for i1, i2, j1, j2 in egaux:
                for i, j in zip(range(i1, i2), range(j1, j2)):
                    if labels_ref[i]:
                        total += 1
                        corrects += int(labels_ref[i] == labels_hyp[j])
            if total:
                score_locuteur = 100.0 * corrects / total

    if score_locuteur is None:
        score_global = score_texte
    else:
        score_global = (1.0 - poids_locuteur) * score_texte + poids_locuteur * score_locuteur
    return {
        "mots_reference": len(tokens_ref),
        "mots_predits": len(tokens_hyp),
        "distance_edition": distance,
        "wer": wer,
        "score_texte": score_texte,
        "score_locuteur": score_locuteur,
        "score_global": score_global,
    }


def ecrire_rapport_scores(candidats, chemin):
    champs = ["tentative", "passe", "profil", "modele", "audio", "beam",
              "mots_reference", "mots_predits", "distance_edition", "WER_pct",
              "score_texte_pct", "score_locuteur_pct", "score_global_pct",
              "duree_passe_s", "retenue"]
    gagnant = max(candidats, key=lambda c: (c["evaluation"]["score_global"],
                                            c["evaluation"]["score_texte"],
                                            -c["evaluation"]["wer"]))
    with open(chemin, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=champs)
        writer.writeheader()
        for c in sorted(candidats, key=lambda x: (x["tentative"], x["passe"])):
            ev = c["evaluation"]
            writer.writerow({
                "tentative": c["tentative"], "passe": c["passe"],
                "profil": c["profil"], "modele": c["modele"],
                "audio": c["audio"], "beam": c["beam"],
                "mots_reference": ev["mots_reference"],
                "mots_predits": ev["mots_predits"],
                "distance_edition": ev["distance_edition"],
                "WER_pct": round(100 * ev["wer"], 3),
                "score_texte_pct": round(ev["score_texte"], 3),
                "score_locuteur_pct": "" if ev["score_locuteur"] is None else round(ev["score_locuteur"], 3),
                "score_global_pct": round(ev["score_global"], 3),
                "duree_passe_s": "" if c.get("duree_s") is None else round(c["duree_s"]),
                "retenue": "OUI" if c is gagnant else "",
            })
    return gagnant


def ecrire_rapport_modeles(candidats, chemin):
    """Agrege les resultats par nom de modele pour aider a choisir celui des prochaines transcriptions."""
    groupes = {}
    for c in candidats:
        e = c["evaluation"]
        g = groupes.setdefault(c["modele"], {
            "scores_texte": [], "scores_global": [], "wers": [], "nombre": 0,
            "meilleur_score": 0.0,
        })
        g["scores_texte"].append(e["score_texte"])
        g["scores_global"].append(e["score_global"])
        g["wers"].append(e["wer"])
        g["nombre"] += 1
        g["meilleur_score"] = max(g["meilleur_score"], e["score_global"])
    lignes = []
    for modele, g in groupes.items():
        lignes.append({
            "modele": modele,
            "nombre_de_passes": g["nombre"],
            "score_texte_moyen_pct": round(float(np.mean(g["scores_texte"])), 3),
            "score_global_moyen_pct": round(float(np.mean(g["scores_global"])), 3),
            "WER_moyen_pct": round(100 * float(np.mean(g["wers"])), 3),
            "meilleur_score_global_pct": round(g["meilleur_score"], 3),
        })
    lignes.sort(key=lambda x: (x["score_global_moyen_pct"], x["score_texte_moyen_pct"]), reverse=True)
    with open(chemin, "w", newline="", encoding="utf-8-sig") as fh:
        champs = ["modele", "nombre_de_passes", "score_texte_moyen_pct",
                  "score_global_moyen_pct", "WER_moyen_pct", "meilleur_score_global_pct"]
        writer = csv.DictWriter(fh, fieldnames=champs)
        writer.writeheader()
        writer.writerows(lignes)
    return lignes


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", help="Fichier audio ou video a transcrire")
    p.add_argument("csv_sortie", help="CSV final : la meilleure transcription selon la reference")
    p.add_argument("--reference", required=True,
                   help="Fichier de transcription de reference (TXT, CSV ou TSV). Utilise seulement pour le score.")
    p.add_argument("--verbeux", action="store_true",
                   help="affiche tous les details a l'ecran (par defaut : barres de progression seulement)")
    p.add_argument("--langue", default="fr")
    p.add_argument("--modele", default="large-v3",
                   help="modele principal : large-v3, ou chemin d'un modele converti (FrWhisper)")
    p.add_argument("--modele2", default=None,
                   help="2e modele pour les passes B (ex. large-v3) ; sinon le modele principal")
    p.add_argument("--seuil-concordance", type=float, default=0.90,
                   help="seuil de concordance note dans le journal (ne stoppe jamais les trois tentatives)")
    p.add_argument("--recommencer", action="store_true",
                   help="ignore les passes deja calculees dans le dossier _essais")
    p.add_argument("--nb-locuteurs", type=int, default=2)
    p.add_argument("--seuil-pause", type=float, default=1.0)
    p.add_argument("--noms", nargs="*", default=[],
                   help="ex. SPEAKER_00=Sujet SPEAKER_01=Psy (etiquettes de la 1re passe)")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--compute", default=None,
                   help="int8_float32 (GPU ancien), float16 (GPU recent)...")
    p.add_argument("--marge-empreinte", type=float, default=0.05)
    p.add_argument("--extra", default="", help="interjections supplementaires")
    p.add_argument("--glossaire", default="",
                   help="mots difficiles separes par des virgules (une vingtaine max)")
    p.add_argument("--sans-empreinte", action="store_true")
    p.add_argument("--sans-lissage", action="store_true")
    p.add_argument("--max-mots-lissage", type=int, default=3)
    p.add_argument("--sans-verif-phrases", action="store_true")
    p.add_argument("--duree-phrase-courte", type=float, default=8.0)
    p.add_argument("--marge-phrase", type=float, default=0.10)
    p.add_argument("--sans-relecture", action="store_true")
    p.add_argument("--duree-min-tour", type=float, default=0.6)
    p.add_argument("--lignes-longues", action="store_true")
    a = p.parse_args()

    t_debut = time.time()
    base = os.path.splitext(a.csv_sortie)[0]
    dossier = base + "_essais"
    os.makedirs(dossier, exist_ok=True)
    initialiser_affichage(a.verbeux, base + "_journal.txt")

    noms = dict(x.split("=", 1) for x in a.noms)
    fillers = set(INTERJECTIONS) | {x.strip().lower() for x in a.extra.split(",")
                                    if x.strip()}
    # Pas de remplacement manuel : la reference sert uniquement a evaluer les resultats.
    regles = []
    reference = charger_reference(a.reference)
    info(f"Reference chargee : {len(reference['tokens'])} mots depuis {reference['path']}")
    if reference.get("has_speakers"):
        info("La reference contient des locuteurs : le score locuteur sera calcule "
             "si --noms correspond aux etiquettes de reference.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    glossaire = ", ".join(x.strip() for x in a.glossaire.split(",") if x.strip())
    ctx = {
        "token": os.environ.get("HF_TOKEN"), "device": device,
        # GPU ancien (GTX 1080 Ti) : pas de float16, et float32 depasse 11 Go
        "compute": a.compute or ("int8_float32" if device == "cuda" else "int8"),
        "fillers": fillers,
        "prompt": PROMPT_VERBATIM + (" " + glossaire + "." if glossaire else ""),
        "prompt_relecture": (glossaire + ".") if glossaire else None,
    }
    nb_t = 3  # trois tentatives obligatoires, sans arret anticipe
    nb_etapes = 4 if a.sans_relecture else 5

    barre = tqdm(total=2 * nb_t + 2, desc="Preparation de l'audio", unit="etape",
                 position=0, file=ECRAN)

    # audio brut + audio normalise en volume ; modele d'empreinte
    audios = {}
    with silencieux():
        with tempfile.TemporaryDirectory() as tmp:
            for cle, filtre in (("brut", None), ("norm", "dynaudnorm")):
                wav = os.path.join(tmp, f"{cle}.wav")
                extraire_audio_filtre(a.video, wav, filtre)
                audios[cle] = whisperx.load_audio(wav)
        emb_modele = None if a.sans_empreinte else charger_modele_empreinte(device, ctx["token"])
    barre.update(1)

    historique, ref = [], None
    candidats = []
    for t in range(1, nb_t + 1):
        profils = (PROFILS[(2 * (t - 1)) % len(PROFILS)],
                   PROFILS[(2 * (t - 1) + 1) % len(PROFILS)])
        info(f"=== Tentative {t}/{nb_t} : passes {profils[0]['nom']} et {profils[1]['nom']} ===")
        passes = []
        for cote, prof in zip("AB", profils):
            barre.set_description(f"Tentative {t}/{nb_t} - passe {prof['nom']}")
            stem = os.path.join(dossier, f"tent{t}_{cote}_{prof['nom']}")
            nom_modele = (a.modele2 if prof["modele"] == "second" and a.modele2
                          else a.modele)
            mots, duree = None, None
            if os.path.exists(stem + ".json") and not a.recommencer:
                mots = charger_mots(stem + ".json")
                if os.path.exists(stem + ".meta.json"):
                    duree = charger_mots(stem + ".meta.json").get("duree_s")
                info(f"  [{prof['nom']}] deja calculee, reprise.")
            else:
                etapes = tqdm(total=nb_etapes, desc=f"   {prof['nom']}", unit="etape",
                              position=1, leave=False, file=ECRAN)
                debut_passe = time.time()
                try:
                    with silencieux():
                        mots = traiter(audios[prof["audio"]], nom_modele, prof, a, ctx,
                                       emb_modele, etapes=etapes)
                    duree = time.time() - debut_passe
                    with open(stem + ".meta.json", "w", encoding="utf-8") as fh:
                        json.dump({"duree_s": duree}, fh)
                except Exception as e:  # noqa: BLE001
                    info(traceback.format_exc())
                    afficher(f"  [{prof['nom']}] ECHEC : {e}  (details dans {os.path.basename(base)}_journal.txt)")
                    liberer_gpu()
                finally:
                    etapes.close()
            if mots is not None:
                if ref is None:
                    ref = mots
                else:
                    aligner_locuteurs(ref, mots)
                sauver_mots(mots, stem + ".json")     # etiquettes deja alignees sur la reference
                ecrire_csv(construire_segments(mots, a.seuil_pause,
                                               not a.lignes_longues, regles),
                           stem + ".csv", noms)
                evaluation = evaluer_candidat(mots, reference, noms)
                candidats.append({
                    "tentative": t, "passe": cote, "profil": prof["nom"],
                    "modele": nom_modele, "audio": prof["audio"],
                    "beam": prof["beam"], "mots": mots,
                    "evaluation": evaluation, "duree_s": duree,
                })
                loc_txt = (" | locuteurs %.1f%%" % evaluation["score_locuteur"]
                           if evaluation["score_locuteur"] is not None else "")
                info(f"  Score reference {prof['nom']} ({nom_modele}) : "
                     f"{evaluation['score_global']:.2f}/100 "
                     f"(WER {evaluation['wer']:.2%}, texte {evaluation['score_texte']:.1f}%{loc_txt})")
            passes.append(mots)
            barre.update(1)

        if passes[0] is None or passes[1] is None:
            info("  Tentative incomplete (une passe a echoue) : ignoree.")
            historique.append({"tentative": t, "profils": profils, "passes": passes,
                               "conc": None})
            continue
        c = concordance(passes[0], passes[1])
        historique.append({"tentative": t, "profils": profils, "passes": passes, "conc": c})
        info(f"  Concordance : texte {c['texte']:.1%} | locuteurs {c['locuteur']:.1%} "
             f"| GLOBALE {c['global']:.1%} (seuil {a.seuil_concordance:.0%})")

    if not candidats:
        barre.close()
        raise SystemExit("Aucune passe n'a abouti : consultez le fichier journal.")

    # Finalisation : choix fonde sur la reference, fichiers de sortie
    barre.set_description("Finalisation des fichiers")
    meilleur_candidat = ecrire_rapport_scores(candidats, base + "_scores_reference.csv")
    classement_modeles = ecrire_rapport_modeles(candidats, base + "_comparaison_modeles.csv")
    retenu = meilleur_candidat["mots"]
    nom_retenu = meilleur_candidat["profil"]
    evaluation = meilleur_candidat["evaluation"]

    segments = construire_segments(retenu, a.seuil_pause, not a.lignes_longues, regles)
    nb_verif = ecrire_csv(segments, a.csv_sortie, noms, base + "_a_verifier.csv")

    # Divergences internes : paire de passes de la tentative gagnante si complete.
    historique_gagnant = next(
        (h for h in historique if h["tentative"] == meilleur_candidat["tentative"]), None)
    if (historique_gagnant and historique_gagnant["conc"] is not None
            and all(m is not None for m in historique_gagnant["passes"])):
        pa, pb = historique_gagnant["passes"]
        nb_div = ecrire_divergences(pa, pb, historique_gagnant["conc"], noms,
                                    base + "_divergences.csv")
    else:
        with open(base + "_divergences.csv", "w", newline="", encoding="utf-8-sig") as fh:
            csv.writer(fh).writerow(["debut_s", "fin_s", "concordance", "passe_A", "passe_B"])
        nb_div = 0

    with open(os.path.join(dossier, "rapport.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["tentative", "passe_A", "passe_B", "concordance_texte",
                    "concordance_locuteurs", "concordance_globale", "retenue_selon_reference"])
        for h in historique:
            c = h["conc"]
            w.writerow([h["tentative"], h["profils"][0]["nom"], h["profils"][1]["nom"],
                        "" if c is None else round(c["texte"], 3),
                        "" if c is None else round(c["locuteur"], 3),
                        "" if c is None else round(c["global"], 3),
                        "oui" if h["tentative"] == meilleur_candidat["tentative"] else ""])
    barre.update(1)
    barre.close()

    # ---- resultat final (seul affichage hors barres)
    duree_totale = time.time() - t_debut
    afficher("")
    afficher("================ RESULTAT FINAL ================")
    for c in sorted(candidats, key=lambda x: (x["tentative"], x["passe"])):
        ev = c["evaluation"]
        marque = "  <== retenue" if c is meilleur_candidat else ""
        afficher(f"Tentative {c['tentative']} passe {c['profil']} ({c['modele']}) : "
                 f"score {ev['score_global']:.2f}/100, WER {ev['wer']:.2%}, "
                 f"duree {'(reprise)' if c.get('duree_s') is None else formater_duree(c['duree_s'])}{marque}")
    afficher("")
    afficher(f"Retenue : tentative {meilleur_candidat['tentative']} / passe {meilleur_candidat['passe']} "
             f"/ profil {nom_retenu} / modele {meilleur_candidat['modele']}")
    afficher(f"Score texte : {evaluation['score_texte']:.2f}/100   WER : {evaluation['wer']:.2%} (plus bas = mieux)")
    if evaluation["score_locuteur"] is not None:
        afficher(f"Score locuteurs : {evaluation['score_locuteur']:.2f}%")
    afficher(f"Score global utilise pour choisir : {evaluation['score_global']:.2f}/100")
    if len(classement_modeles) > 1:
        top = classement_modeles[0]
        afficher(f"Modele le mieux classe en moyenne : {top['modele']} "
                 f"({top['score_global_moyen_pct']:.2f}/100 sur {top['nombre_de_passes']} passes)")
    afficher("")
    afficher(f"Fichier final   : {a.csv_sortie} ({len(segments)} lignes)")
    afficher(f"A verifier      : {base}_a_verifier.csv ({nb_verif} lignes)")
    afficher(f"Divergences     : {base}_divergences.csv ({nb_div} passages de 30 s)")
    afficher(f"Scores          : {base}_scores_reference.csv")
    afficher(f"Journal complet : {base}_journal.txt")
    afficher("")
    durees = [c["duree_s"] for c in candidats if c.get("duree_s") is not None]
    if durees:
        afficher(f"Temps de calcul cumule des passes : {formater_duree(sum(durees))}")
    afficher(f"DUREE TOTALE DE LA TRANSCRIPTION : {formater_duree(duree_totale)}")


if __name__ == "__main__":
    main()
