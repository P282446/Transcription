# Transcription automatique d'entretiens avec évaluation par référence

## 1. Présentation du projet

Ce programme permet de transcrire automatiquement un entretien audio ou vidéo, puis d'évaluer les résultats obtenus en les comparant à une transcription de référence.

Il utilise **WhisperX**, un modèle de transcription configurable et un système de diarisation pour identifier les locuteurs. Plusieurs configurations sont testées afin de sélectionner automatiquement la transcription qui obtient le meilleur score par rapport à la référence.

Le programme est conçu pour les entretiens comportant deux locuteurs, par exemple un psychologue (`Psy`) et un sujet (`Sujet`).

## 2. Principe de fonctionnement

Le programme réalise systématiquement **trois tentatives, comprenant chacune deux passes de transcription**, soit jusqu'à six passes.

Chaque passe utilise une configuration différente afin de comparer plusieurs résultats. Les configurations peuvent varier selon :

- Le signal audio utilisé : audio brut ou normalisé en volume.
- Les paramètres de détection de la parole (VAD).
- La précision du décodage.
- La graine aléatoire utilisée pour certains traitements.
- Le modèle de transcription employé, notamment lorsque deux modèles sont configurés.

### Étapes du pipeline

1. **Préparation de l'audio** : extraction de la piste audio et création de deux versions, brute et normalisée.
2. **Diarisation** : détection des tours de parole et identification des locuteurs.
3. **Transcription** : reconnaissance automatique des paroles.
4. **Alignement temporel** : association des mots à leurs intervalles temporels.
5. **Post-traitement** : vérification des attributions de locuteurs, traitement des interjections et relecture automatique de certains passages.
6. **Évaluation** : comparaison de chaque transcription avec la transcription de référence.
7. **Sélection** : conservation du résultat qui obtient le meilleur score par rapport à la référence.
8. **Exportation** : génération de la transcription finale, des scores et des rapports de contrôle.

### Rôle de la transcription de référence

Le fichier de référence sert exclusivement à **évaluer les résultats et à sélectionner la meilleure transcription**.

Il n'est pas utilisé pour guider la reconnaissance vocale, remplacer automatiquement les mots reconnus ou compléter les phrases générées par les modèles.

## 3. Évaluation des transcriptions

Le critère principal est le taux d'erreur par mot, appelé *Word Error Rate* (WER).

Il est calculé à partir du nombre de substitutions, de suppressions et d'insertions :

\[
\mathrm{WER} = \frac{S + D + I}{N}
\]

avec :

- \(S\) : nombre de substitutions ;
- \(D\) : nombre de suppressions ;
- \(I\) : nombre d'insertions ;
- \(N\) : nombre de mots dans la transcription de référence.

Un score de qualité est ensuite calculé :

\[
\mathrm{Score} = \max(0,\;100 \times (1-\mathrm{WER}))
\]

Un score plus élevé correspond à une transcription textuellement plus proche de la référence.

Lorsque les étiquettes de locuteurs sont disponibles dans la référence et peuvent être mises en correspondance avec celles du programme, une évaluation complémentaire de l'attribution des locuteurs peut également être réalisée.

Les scores sont utilisés pour choisir la meilleure transcription parmi les résultats produits. **Le meilleur résultat sur un entretien ne garantit cependant pas que la même configuration sera la meilleure sur tous les autres entretiens.**

## 4. Affichage et suivi de l'exécution

Pour faciliter le suivi des traitements longs, l'affichage dans le terminal est limité par défaut :

- Une barre de progression générale indique l'avancement du programme.
- Une barre secondaire indique les étapes de la passe en cours.
- Les messages détaillés, avertissements et informations de traitement sont enregistrés dans un fichier journal.

À la fin de l'exécution, le programme affiche le bilan des tentatives, les scores obtenus et la durée totale du traitement.

L'option `--verbeux`, lorsqu'elle est disponible dans la version utilisée, permet d'afficher davantage de détails dans le terminal.

## 5. Fichiers générés

Pour un fichier de sortie nommé `SORTIE.csv`, le programme génère les fichiers suivants :

| Fichier | Description |
|---|---|
| `SORTIE.csv` | Transcription finale ayant obtenu le meilleur score face à la référence. |
| `SORTIE_a_verifier.csv` | Segments signalés comme douteux ou nécessitant une vérification. |
| `SORTIE_divergences.csv` | Passages pour lesquels les deux passes de la tentative retenue présentent des différences. |
| `SORTIE_scores_reference.csv` | Scores de chaque passe calculés par comparaison avec la transcription de référence. |
| `SORTIE_comparaison_modeles.csv` | Comparaison des performances moyennes des modèles testés. |
| `SORTIE_journal.txt` | Journal détaillé de l'exécution. |
| `SORTIE_essais/` | Résultats intermédiaires de chaque passe, fichiers CSV et JSON, et rapport des tentatives. |

Ces fichiers permettent de comparer les modèles, de conserver les résultats intermédiaires et d'identifier les passages pour lesquels la transcription est moins fiable.

## 6. Prérequis

Le programme nécessite un environnement Python compatible avec les bibliothèques utilisées par le pipeline, notamment :

- WhisperX ;
- PyTorch ;
- NumPy ;
- `tqdm` ;
- `faster-whisper` ;
- `pyannote.audio` pour la diarisation et les fonctionnalités associées ;
- FFmpeg pour l'extraction audio.

Un accès aux modèles concernés sur Hugging Face peut également être nécessaire.

## 7. Installation

### 7.1. Créer et activer un environnement virtuel

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 7.2. Installer les dépendances

Installez les bibliothèques nécessaires dans l'environnement virtuel, selon les versions compatibles avec votre configuration.

FFmpeg doit également être installé et disponible dans le `PATH` du système.

### 7.3. Configurer le jeton Hugging Face

Si les modèles utilisés nécessitent une authentification, configurez votre jeton dans une variable d'environnement :

```bash
export HF_TOKEN=hf_votre_jeton
```

Remplacez `hf_votre_jeton` par votre propre jeton. Ne publiez jamais ce jeton dans le dépôt Git.

## 8. Utilisation

### 8.1. Exemple de commande

```bash
python transcription.py entretien.m4a sortie.csv \
    --reference vraie_transcription.csv \
    --langue fr \
    --noms SPEAKER_00=Sujet SPEAKER_01=Psy \
    --modele ~/models/frwhisper-ct2 \
    --modele2 large-v3
```

### 8.2. Description des paramètres

| Paramètre | Description |
|---|---|
| `entretien.m4a` | Fichier audio ou vidéo à transcrire. |
| `sortie.csv` | Chemin du fichier de transcription finale. |
| `--reference` | Chemin du fichier contenant la transcription de référence. |
| `--langue fr` | Langue de transcription : français. |
| `--noms` | Correspondance entre les identifiants des locuteurs et leurs noms. |
| `--modele` | Modèle principal de transcription. |
| `--modele2` | Second modèle utilisé dans les passes qui lui sont affectées. |

Les chemins doivent être adaptés à l'organisation réelle des fichiers sur la machine.

## 9. Reproductibilité et recommandations

- Conserver la transcription de référence utilisée pour chaque évaluation.
- Vérifier que les transcriptions de référence sont correctement alignées avec les enregistrements audio.
- Utiliser la même convention de nommage des locuteurs dans les différents fichiers.
- Conserver les fichiers intermédiaires et les rapports pour pouvoir comparer les résultats.
- Évaluer plusieurs entretiens de référence pour identifier le modèle le plus performant sur l'ensemble du corpus.
- Vérifier les passages signalés comme douteux avant d'utiliser les transcriptions dans une analyse scientifique.

## 10. Objectif du pipeline

L'objectif est de **réduire les erreurs de transcription et d'identifier les modèles les plus performants sur un corpus d'entretiens** grâce à une évaluation quantitative par rapport à des transcriptions de référence.

Le processus est automatisé : les résultats sont comparés, les scores sont calculés et la meilleure transcription candidate est sélectionnée sans remplacement manuel des paroles à partir du fichier de référence.
