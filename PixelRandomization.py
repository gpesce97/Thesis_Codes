import os
import time as tm
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def get_cum_distance(wafer: np.ndarray, diff) -> int:
    N_KIDs = wafer.max() + 1
    distance = []
    mask_value = -2 * diff

    for i in range(wafer.shape[0]):
        for j in range(wafer.shape[1]):
            if wafer[i, j] != mask_value:
                for k in range(max(0, wafer[i, j] - diff), min(wafer[i, j] + diff, N_KIDs)):
                    if wafer[i,j] == k: continue
                    ii, jj = np.where(wafer == k)
                    distance.append((((ii - i) ** 2 + (jj - j) ** 2) ** 0.5)[0])

    return sum(distance)


def plot_wafer(id_conf: int, wafer: np.ndarray, diff: int = -8, savedir: str = '.'):

    mask_value = -2 * diff
    distance = get_cum_distance(wafer, diff)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), tight_layout=True)

    offset = 0.25
    vertices = []
    for v in [(0, 3), (0 - offset, 9), (3, 12), (9, 12), (12, 9), (12, 3), (9, 0), (3, 0)]:
        vertices.append((v[0] / wafer.shape[0], v[1] / wafer.shape[1]))

    octagon = patches.Polygon(vertices, closed=True, edgecolor='black', facecolor='whitesmoke')

    ax.set(facecolor='white', label=False)
    ax.set_axis_off()
    ax.add_patch(octagon)

    for x in range(wafer.shape[0]):
        for y in range(wafer.shape[1]):

            if wafer[x, y] != mask_value:
                bbox = dict(boxstyle=patches.BoxStyle("Round", pad=0.4),
                            edgecolor='orange',
                            facecolor='white')

                ax.text(**dict(x=x / wafer.shape[0], y=y / wafer.shape[1],
                               s=f"{wafer[x][y] + 1:03}", size=10, rotation='horizontal', bbox=bbox))

    ax.set_title(f'cumulative distance: {distance: .2f} | min diff: {-mask_value // 2}')
    ax.set_aspect('equal')

    fig.savefig(os.path.join(savedir, f'{id_conf:03}_wafer_conf'), dpi=600)


def create_configuration(diff: int = 2):

    N = 13  # ordine della matrice
    N_KIDs = 145  # numero di KIDs da posizionare
    mask_value = -2 * diff # maschera gli angoli della matrice quadrata per ottenere un ottagono
    offset = 10  # necessario per creare l'ottagono
    # diff = 4  # differenza minima tra gli indici di un KID e i suoi vicini
    fill_value = N_KIDs + diff  # necessario per identificare e riempire le celle dell'ottagono
    wafer = np.full(shape=(N, N), dtype=int, fill_value=fill_value)

    # generatore numeri casuali
    rng = np.random.default_rng(123456)

    # maschero le celle per ottenere l'ottagono
    for x in range(3):
        for y in range(3 - x):
            # pongo le celle invalide a -diff, in modo che la somma con diff generi 0
            wafer[x, y] = wafer[y + offset + x, x] = wafer[x, y + x + offset] = wafer[N - x - 1, N - y - 1] = mask_value

    kids = list(range(N_KIDs))
    for i in range(N):
        for j in range(N):
            # considero solamente le celle appartenenti all'ottangono
            # cioe' quelle che hanno come valore iniziale `fill_value`
            if wafer[i][j] == fill_value:

                # considero la matrice (quadrata se possibile, altrimenti rettangolare)
                # che ha come elemento centrale l'indice del KID appena estratto
                sub = wafer[max(0, i - 1): min(i + 2, len(wafer)),
                      max(j - 1, 0): min(j + 2, wafer.shape[1])]

                # min_kid = min(kids)
                #
                # # se non ci sono piu' kid che massimizzano la distanza, esci
                # if np.all(np.abs(sub - min_kid) < diff):
                #     return []

                # estraggo l'indice di un KID
                kid = rng.choice(kids)

                # estrai un nuovo kid fino a quando tutti ii vicini (in termini di indice) non sono distanti
                # dal KID selezionato almeno 'diff' valori
                while not np.all(np.abs(sub - kid) >= diff):
                    kid = rng.choice(kids)

                # posiziono il kid nell'ottagono e lo rimuovo dalle possibil scelte
                wafer[i][j] = kid
                kids.pop(kids.index(kid))

    return wafer


if __name__ == '__main__':

    diff = 2
    N_conf = 20

    res = []
    configurations = []

    start = tm.perf_counter()

    # with mp.Pool() as pool:
    #     configurations = pool.map(create_configuration, [diff] * N_conf, chunksize=max(mp.cpu_count() // N_conf, 1))

    with mp.Pool() as pool:
        for _ in range(N_conf):
            res.append(pool.apply_async(create_configuration, args=(diff,)))

        for r in res:
            try:
                configurations.append(r.get(timeout=0.1))
            except mp.TimeoutError:
                pass

    configurations.sort(key=lambda x: get_cum_distance(x, diff), reverse=True)
    # ordina le configurazioni in base alla maggior distanza comulativa (decrescente)
    end = tm.perf_counter()
    execution_time = f"Time to find {len(configurations)} conf: {end - start:.2f}[s]"
    print(execution_time)

    savedir = './wafer_confgs'
    os.makedirs(savedir, exist_ok=True)

    with mp.Pool() as pool:
        res = pool.starmap(plot_wafer,
                           zip(range(len(configurations)),  # id della configurazione
                               configurations,  # configurazioni
                               len(configurations) * [diff],  # minimal diff
                               len(configurations) * [savedir]),  # cartella in cui salvare i plot
                           chunksize=max(1, len(configurations) // mp.cpu_count()))  # carico di ogni cpu

"""
configurations =                   [c0, c1, c2, ... , cn]
range(len(configurations)) =       [0, 1, 2, 3, 4, ..., N]
len(configurations) * [-8] =       [-8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8]
len(configurations) * [savedir]) = [savedir, savedir, savedir, savedir, savedir, savedir, savedir, savedir, savedir, savedir]
"""
