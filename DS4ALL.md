# ds4all

Fork di [antirez/ds4](https://github.com/antirez/ds4) con un obiettivo preciso:
far girare DeepSeek V4 Flash (81 GB di pesi) su un PC consumer con una GPU da
8 GB di VRAM.

## Cosa cambia rispetto a ds4

Il branch `hybrid-moe` aggiunge uno **split funzionale CPU/GPU degli esperti
MoE**: la parte sempre-accesa del modello (attention, router, esperti shared)
sta in VRAM, gli esperti routed restano fuori e vengono serviti a tre livelli —
VRAM per i più caldi, RAM pinned, SSD per la coda fredda. Il criterio di
partizione è il ruolo funzionale del tensore, non la posizione del layer.

La misura di riferimento (RTX 4060 8 GB, 32 GB RAM): prefill +45% rispetto al
solo CPU, decode limitato dalla banda del tier più lento toccato. Nota di
correttezza: l'output GPU non è bit-identico alla CPU (drift fast-math
benigno) — non validare con confronti di testo esatto.

## Stato e destinazione

Progetto autonomo, nato nel workspace quasar come banco didattico: il valore è
capire il meccanismo, non competere con `llama.cpp -ot` (che resta la via
consigliata per l'uso quotidiano). Repo privato per ora; la storia del fork e
il remote `upstream` restano agganciati ad antirez/ds4 per sincronizzare.

Documentazione di dettaglio nel workspace d'origine: architettura dello split
e del tiering, benchmark e diario in `quasar/docs/` (ARCHITETTURA_DS4_TIERING,
ROADMAP, ARTICOLO_DIVULGATIVO).
