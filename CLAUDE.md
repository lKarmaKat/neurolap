# Neurolap : jeu de voitures 2D pour entraînement d'IA

## Objectif
Jeu de voitures 2D solo (pas de multijoueur) conçu pour qu'un modèle puisse le piloter.
Phase actuelle : **le jeu uniquement** (pas d'entraînement). Phase suivante : neuroévolution, puis éventuellement RL.
Le jeu doit donc exposer une interface propre pour un agent IA dès le départ.

## Architecture (3 couches séparées)
1. `sim/` (Python, NumPy) : source de vérité. Physique, circuit, capteurs, score. **Headless**, déterministe (seed), aucune dépendance au rendu.
2. `server/` (Python) : sert l'état de la simulation au navigateur (WebSocket) et reçoit les commandes clavier en mode manuel.
3. `web/` (TypeScript + Canvas) : rendu seul. Aucune logique de jeu. Affiche voiture, circuit, rayons des capteurs, score.

Règle : la simulation ne doit jamais importer le serveur ni le rendu. L'entraînement futur n'utilisera que `sim/`.

## Interface IA (à respecter dès la phase 1)
API de type Gymnasium, sans dépendance obligatoire :
- `env.reset(seed=None) -> observation, info`
- `env.step(action) -> observation, reward, terminated, truncated, info`
- **Action** : `[steering, throttle]`, floats dans [-1, 1]. steering : -1 gauche, 0 droit, 1 droite. throttle : -1 reculer, 0 rien, 1 avancer. Valeurs clippées par l'env.
- **Observation** : vecteur normalisé : distances des rayons aux murs (7 rayons de -90° à +90°, valeurs dans [0, 1]) + vitesse + vitesse angulaire.
- **Reward / score** : progression le long de la ligne centrale du circuit (pas la distance brute). Reculer ne rapporte rien. `info["completion"]` = taux de complétion du circuit dans [0, 1].
- **Fin d'épisode** : collision mur (`terminated`), circuit complété (`terminated`), temps max (`truncated`).
- Pas de temps fixe (dt constant), indépendant de l'affichage.
Rester compatible avec un futur wrapper Gymnasium sans le faire maintenant.

## Circuits
- Générateur procédural avec seed (courbe fermée lissée -> murs gauche/droit + ligne centrale).
- Objectif : éviter le surapprentissage sur un seul tracé.
- Éditeur de circuits dans le navigateur : à faire **après** que le générateur et l'API fonctionnent, pas avant.

## Périmètre de la phase 1
Inclus : sim, capteurs, score, générateur de circuits, rendu navigateur, mode manuel clavier, tests de la sim.
Exclus : entraînement, multijoueur, sons, menus, graphismes élaborés, sauvegarde de comptes.
Physique volontairement simple (voiture type vélo cinématique, pas de simulation réaliste).

## Commandes
Depuis la racine du projet (Windows, venv dans `.venv`) :
- Installer : `python -m venv .venv` puis `.venv\Scripts\python -m pip install -e ".[dev]"` (ou `pip install numpy pytest ruff`)
- Tests : `.venv\Scripts\python -m pytest`
- Lint : `.venv\Scripts\python -m ruff check .`
- Serveur : _à compléter (incrément 6)_
- Web : _à compléter (incrément 7)_

## Économie de tokens (priorité du projet)

### Modèles et effort
- Agent principal : **Sonnet, effort moyen**. C'est le défaut pour coder.
- **Opus : uniquement si je le demande** ou pour une décision d'architecture bloquante. Dans ce cas, le dire avant de l'utiliser.
- Sous-agents d'exploration (recherche de fichiers, lecture, tâches mécaniques) : **Haiku ou Sonnet effort bas**.
- Ne jamais lancer un modèle plus puissant que nécessaire. Si un modèle plus léger suffit, l'utiliser.

### Sous-agents
- Pas de sous-agent par défaut. Travailler directement sauf si la tâche est volumineuse (balayage de nombreux fichiers) ou réellement parallélisable.
- Un sous-agent reçoit un prompt court et précis, retourne un résumé court (pas de dump de fichiers).
- Ne pas lancer plusieurs sous-agents pour la même question.

### Règles anti-boucle
- Maximum **2 tentatives** pour corriger une erreur. Après 2 échecs : s'arrêter, résumer le blocage, me demander.
- Ne pas relire un fichier déjà lu dans la session sauf s'il a changé.
- Lire uniquement les parties nécessaires d'un fichier (offset/limit), pas tout par réflexe.
- Pas de refonte, de refactor ni de renommage non demandés.
- Pas de tests ou vérifications répétés si rien n'a changé.
- Ne pas poser une question dont la réponse est déjà dans ce fichier ou dans la conversation.
- Une tâche = un objectif. Ne pas élargir le périmètre sans me le dire.

### Style de réponse
- Direct, sans remplissage ni récapitulatif des étapes. Concis sur l'action, mais pas au détriment des explications demandées dans la section Pédagogie.
- Fin de tâche : ce qui a été fait, ce qui reste (1 à 3 lignes), une prochaine étape si pertinente.
- Réponses complètes et structurées.

## Pédagogie (priorité équivalente à l'économie de tokens)
Je veux **apprendre** : ce projet est aussi un support de formation (simulation, capteurs, neuroévolution, RL). Je suis ingénieur et développeur (front, TypeScript, Svelte), mais pas spécialiste de l'IA ni de la simulation physique.
- **Expliquer chaque choix** : pourquoi cette solution, quelles alternatives existaient, ce qu'on gagne et ce qu'on perd.
- **Expliquer les impacts et implications** : effet sur l'entraînement futur, la performance, la maintenance, le surapprentissage, etc.
- **Expliquer le fonctionnement** : comment marche chaque brique (modèle de voiture, raycasts, calcul de progression, déterminisme, boucle de simulation, protocole client/serveur…), avec un exemple concret ou un petit schéma quand ça aide.
- Utiliser le vocabulaire correct et définir chaque terme technique ou d'IA à sa première apparition.
- Faire le lien avec ce que je connais déjà (TypeScript, front, architecture logicielle) quand c'est utile.
- Si je pose une question, répondre à la question avec la profondeur nécessaire, sans me renvoyer à la documentation.

Format pour ne pas gaspiller de tokens :
- Les explications vont **aux points de décision et pour les notions nouvelles**, pas à chaque ligne de code.
- Concises mais complètes : structurées (court paragraphe ou liste), sans remplissage ni répétition d'une explication déjà donnée.
- Les explications détaillées vont dans la réponse à l'utilisateur, pas dans des commentaires de code verbeux. Les commentaires du code restent courts et utiles.
- Si une notion est vaste, donner l'essentiel et proposer d'approfondir plutôt que tout dérouler.
- Cette exigence pédagogique prime sur la concision : ne jamais supprimer une explication utile à ma compréhension pour économiser des tokens. On économise sur les modèles, les sous-agents, les relectures et les boucles, pas sur la qualité de l'enseignement.

## Validation obligatoire avant toute modification
**Aucune modification sans mon feu vert explicite.** L'agent ne prend jamais l'initiative de modifier quoi que ce soit.
- Avant d'appliquer un changement : décrire ce qu'il compte modifier (fichiers concernés, nature du changement, raison), puis **attendre que je dise explicitement de l'appliquer** (« applique », « vas-y », etc.).
- Un silence, une question de ma part ou un « ok, je vois » ne valent pas accord.
- Cela couvre : création, modification, suppression ou renommage de fichiers, installation de dépendances, changements de configuration, commandes qui écrivent quelque chose sur le disque ou changent l'état du projet, commits.
- Lecture, exploration et analyse sont autorisées sans demander. Lancer les tests existants aussi, tant que cela ne modifie rien au projet.
- Pas de « petite correction au passage » : un bug ou un défaut repéré hors du sujet est signalé, pas corrigé.
- Si le changement dépasse ce qui a été validé (autres fichiers, autre approche), s'arrêter et redemander.
- Une validation vaut pour le changement décrit, pas pour la suite : chaque nouvelle modification demande un nouvel accord.

## Méthode de travail
1. Proposer un plan court (étapes numérotées) avant de coder, avec les choix principaux expliqués, et attendre mon accord explicite avant d'écrire du code (voir « Validation obligatoire »).
2. Avancer par petits incréments testables : sim minimale -> capteurs -> score -> générateur -> serveur -> rendu -> mode manuel.
3. Chaque incrément : tests unitaires sur `sim/` (déterminisme avec seed, collisions, calcul de progression, bornes des actions).
4. Valider avec moi avant de passer à l'étape suivante.

## Git
- Remote : `origin` = `https://github.com/lKarmaKat/neurolap.git`. Branche principale : `master`.
- Commits **réguliers** : un commit par incrément validé (ou plus fin si pertinent), message court en français à l'impératif. Tests et `ruff` au vert avant de committer.
- `push` après chaque commit (ou groupe de commits cohérent).
- Travail direct sur `master` pour les petits changements. **Branche topique** (`feat/...`, `fix/...`) pour un changement plus gros ou expérimental, puis fusion dans `master` une fois validé.
- Ne jamais committer `.venv/`, secrets, ni fichiers générés (voir `.gitignore`). Pas de `push --force` ni de réécriture d'historique sans accord.
- Cette section vaut accord permanent pour `commit` et `push` dans ce cadre ; elle prime sur la mention « commits » de la section Validation. Tout le reste de la validation reste en vigueur.

## Conventions
- Code et commentaires : noms de code en anglais, commentaires et docs en français.
- Python typé (type hints), `ruff` pour le lint, `pytest` pour les tests.
- TypeScript strict côté web.
- Pas de dépendance ajoutée sans justification (garder le projet léger).
