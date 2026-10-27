# Transcription automatique d'entretiens audio et vidéo

## 1. Présentation du projet

Ce projet propose un pipeline de transcription automatique d'entretiens audio ou vidéo en français. Il est conçu en particulier pour les entretiens comportant deux interlocuteurs, par exemple un psychologue (`Psy`) et un sujet (`Sujet`).

Le programme s'appuie sur WhisperX, Faster-Whisper et pyannote pour réaliser la transcription, l'alignement temporel des mots et l'identification des locuteurs (diarisation).

Il répond à deux besoins :

- **Transcription de nouveaux entretiens** : produire une transcription automatique à partir d'un fichier audio ou vidéo, même lorsqu'aucune transcription de référence n'existe.
- **Évaluation des modèles** : comparer les transcriptions automatiques à une transcription de référence afin de mesurer leurs performances et d'identifier les configurations les plus adaptées au corpus.

## 2. Principe de fonctionnement

Le programme effectue trois tentatives de transcription, chacune comprenant deux passes, soit jusqu'à six passes au total.

Les passes utilisent des configurations différentes afin de comparer plusieurs résultats. Les différences peuvent notamment concerner :

- Le signal audio utilisé : audio brut ou normalisé en volume.
- Les paramètres de détection de la parole (VAD).
- Les paramètres de décodage.
- Le modèle de transcription utilisé.
- Certains paramètres aléatoires des traitements.

### Étapes du pipeline

1. **Extraction audio** : extraction de la piste sonore à partir du fichier audio ou vidéo, puis préparation de versions adaptées au traitement.
2. **Diarisation** : détection des tours de parole et attribution des segments aux locuteurs.
3. **Transcription automatique** : reconnaissance des paroles à l'aide du ou des modèles configurés.
4. **Alignement temporel** : association des mots à leurs intervalles temporels.
5. **Post-traitement** : vérification de certaines attributions de locuteurs, traitement des interjections et relecture automatique de passages ciblés.
6. **Évaluation et sélection** : comparaison des candidats selon les critères disponibles, puis conservation du résultat retenu.
7. **Exportation** : production de la transcription finale et des rapports de traitement.

Les opérations sont automatisées. Les résultats restent néanmoins susceptibles de contenir des erreurs de transcription, de ponctuation ou d'attribution des locuteurs.

## 3. Prérequis

Le programme nécessite :

- Python 3.12, dans l'environnement utilisé pour le développement du pipeline.
- FFmpeg pour l'extraction audio.
- WhisperX et Faster-Whisper pour la transcription et l'alignement.
- PyTorch pour l'exécution des modèles.
- pyannote.audio pour la diarisation et les fonctionnalités vocales associées.
- NumPy et `tqdm` pour les traitements numériques et le suivi de l'exécution.

Un accès aux modèles concernés sur Hugging Face peut également être nécessaire.

Les versions des bibliothèques doivent être compatibles entre elles et avec le processeur ou le GPU utilisé.

## 4. Installation

### 4.1. Créer un environnement virtuel

Depuis le répertoire du projet, créez un environnement Python :

```bash
python3.12 -m venv .venv
```

Activez-le :

```bash
source .venv/bin/activate
```

Vérifiez que Python pointe bien vers l'environnement virtuel :

```bash
which python
python --version
```

### 4.2. Installer les dépendances

Installez les bibliothèques nécessaires dans l'environnement activé, en respectant les versions compatibles avec le projet.

FFmpeg doit également être disponible dans le `PATH` du système.

### 4.3. Configurer l'accès aux modèles

Certains modèles nécessitent un jeton Hugging Face. Si nécessaire, configurez-le dans une variable d'environnement :

```bash
export HF_TOKEN=hf_votre_jeton
```

Remplacez `hf_votre_jeton` par votre propre jeton. Ne publiez jamais cette valeur dans le dépôt Git.

## 5. Utilisation principale : transcrire un nouvel entretien sans référence

Cette utilisation est prévue pour les entretiens qui n'ont jamais été transcrits. **Aucun fichier de transcription de référence n'est nécessaire.**

### 5.1. Préparer le fichier à transcrire

Le programme accepte un chemin vers un fichier audio ou vidéo pris en charge par FFmpeg.

Exemples de formats :
- Audio : `.wav`, `.mp3`, `.m4a`.
- Vidéo : `.mp4`, `.mov`, et autres formats lisibles par FFmpeg.

### 5.2. Lancer la transcription automatique

Exemple de commande :

```bash
python transcription.py \
    entretien.m4a \
    transcription.csv \
    --langue fr \
    --noms SPEAKER_00=Psy SPEAKER_01=Sujet \
    --modele ~/models/frwhisper-ct2 \
    --modele2 large-v3
```

Les chemins doivent être adaptés à votre installation.

**Description des paramètres :**

| Paramètre | Description |
|---|---|
| `entretien.m4a` | Fichier audio ou vidéo à transcrire. |
| `transcription.csv` | Chemin du fichier de transcription finale. |
| `--langue fr` | Langue de l'entretien : français. |
| `--noms` | Correspondance entre les identifiants des locuteurs et leurs noms. |
| `--modele` | Modèle principal de transcription. |
| `--modele2` | Second modèle utilisé dans les passes qui lui sont affectées. |

Les identifiants `SPEAKER_00` et `SPEAKER_01` doivent correspondre aux étiquettes réellement produites par la diarisation. La correspondance avec `Sujet` et `Psy` doit être vérifiée pour chaque entretien.

### 5.3. Comprendre les résultats sans référence

Sans option `--reference`, le programme ne peut pas calculer le taux d'erreur par mot par rapport à une transcription de vérité terrain.

Il peut produire une transcription automatiquement sélectionnée à partir des mécanismes internes de comparaison ou de sélection implémentés dans le script. Cette sélection ne constitue toutefois pas une mesure objective de la précision réelle.

Les segments signalés comme incertains doivent être examinés avec une attention particulière, notamment lorsque les deux modèles divergent ou lorsque l'attribution du locuteur est ambiguë.

## 6. Utilisation complémentaire : évaluer les modèles avec une référence

Lorsqu'une transcription fiable existe déjà pour un entretien, elle peut servir à comparer les résultats automatiques et à évaluer les performances des modèles.

### 6.1. Exemple de commande

```bash
python transcription.py \
    entretien.m4a \
    transcription_evaluee.csv \
    --reference vraie_transcription.csv \
    --langue fr \
    --noms SPEAKER_00=Psy  SPEAKER_01=Sujet \
    --modele ~/models/frwhisper-ct2 \
    --modele2 large-v3
```

Le fichier fourni avec `--reference` doit être préparé dans un format compatible avec le programme et contenir le texte de référence. Si l'évaluation des locuteurs est souhaitée, les étiquettes des locuteurs doivent également pouvoir être mises en correspondance.

### 6.2. Calcul du taux d'erreur

Le principal indicateur est le WER (*Word Error Rate*), ou taux d'erreur par mot :

\[
\mathrm{WER}=\frac{S+D+I}{N}
\]

où :

- \(S\) correspond aux substitutions ;
- \(D\) correspond aux suppressions ;
- \(I\) correspond aux insertions ;
- \(N\) est le nombre de mots de la référence.

Un score de lecture sur 100 peut être calculé à partir du WER :

\[
\mathrm{Score}=\max(0,\;100(1-\mathrm{WER}))
\]

Un score plus élevé correspond à une transcription plus proche de la référence selon cette mesure.

Le programme peut aussi calculer des indicateurs complémentaires, notamment pour les locuteurs, lorsque les données nécessaires sont disponibles.

**Attention :** la transcription de référence sert à évaluer et à sélectionner les résultats. Elle ne doit pas être confondue avec une correction manuelle du texte généré.

### 6.3. Choisir un modèle pour les entretiens futurs

L'évaluation peut être répétée sur plusieurs entretiens disposant de transcriptions de référence.

Les rapports permettent alors de comparer les configurations et d'identifier celles qui présentent les meilleures performances sur le corpus évalué.

Il est recommandé de privilégier une évaluation sur plusieurs entretiens représentatifs plutôt que de choisir un modèle à partir d'un seul fichier.

## 7. Fichiers de sortie

Les fichiers produits dépendent du nom de sortie fourni et des options activées dans le programme.

Pour une sortie nommée `SORTIE.csv`, les principaux résultats peuvent comprendre :

| Fichier | Description |
|---|---|
| `SORTIE.csv` | Transcription finale retenue par le programme. |
| `SORTIE_a_verifier.csv` | Segments signalés comme douteux ou nécessitant une vérification. |
| `SORTIE_divergences.csv` | Passages présentant des divergences entre les passes comparées. |
| `SORTIE_scores_reference.csv` | Scores calculés par rapport à une transcription de référence, lorsque cette option est utilisée. |
| `SORTIE_comparaison_modeles.csv` | Rapport de comparaison des modèles, lorsque l'évaluation correspondante est activée. |
| `SORTIE_journal.txt` | Journal détaillé de l'exécution. |
| `SORTIE_essais/` | Fichiers intermédiaires des tentatives et rapports associés. |

Les rapports d'évaluation nécessitent une transcription de référence. En mode transcription seule, les fichiers qui dépendent de cette référence peuvent ne pas être produits.

## 8. Affichage et suivi de l'exécution

Par défaut, le programme présente une interface de suivi concise avec des barres de progression.

Elles permettent de suivre la préparation du fichier et les différentes passes de transcription.

Les messages détaillés, les avertissements et les informations des étapes de traitement sont consignés dans le journal de sortie. Le programme affiche également un bilan à la fin, comprenant les résultats disponibles et la durée totale d'exécution.

L'exécution peut être longue, car plusieurs passes sont réalisées pour chaque entretien.

## 9. Recommandations

- Tester le programme sur un extrait audio avant de traiter un entretien complet.
- Vérifier que le fichier audio ou vidéo est lisible et contient une piste sonore exploitable.
- Vérifier la correspondance entre les identifiants de diarisation et les noms des locuteurs.
- Conserver les fichiers intermédiaires utiles pour comparer les résultats.
- Utiliser des entretiens de référence représentatifs pour évaluer les modèles avant de les utiliser sur de nouveaux entretiens.
- Examiner les segments signalés comme incertains avant toute analyse scientifique.
- Ne pas publier de fichiers audio, de transcriptions contenant des données personnelles ou de jetons d'accès dans le dépôt Git.

## 10. Objectif du projet

L'objectif de ce pipeline est de faciliter la transcription d'entretiens audio et vidéo, y compris lorsque aucune transcription préalable n'existe.

Le programme automatise la reconnaissance vocale, l'alignement temporel et la diarisation, et fournit des outils d'évaluation lorsqu'une transcription de référence est disponible.

Il vise ainsi à réduire le travail manuel tout en permettant d'étudier quantitativement les performances des modèles de transcription et leur adéquation au corpus étudié.
