
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

from optimizador_colonia_de_hormigas import AntColonyOptimizer
from costos import matriz_costos, leer_peajes, leer_capitales

import imageio_ffmpeg
import matplotlib as mpl


# =====================================
# CREAR ANIMACIÓN
# =====================================

def crear_animacion(
    optimizer,
    ubicaciones,
    espana,
    version,
    intervalo=5,
    fps=5,
    exportar_gif=True,
    exportar_mp4=True,
):

    rutas = optimizer.historial_rutas
    costos = optimizer.historial_costos

    # Seleccionar una muestra de las iteraciones
    indices = list(range(0, len(rutas), intervalo))

    # Asegurar que se muestre la solución final
    if indices[-1] != len(rutas) - 1:
        indices.append(len(rutas) - 1)

    fig, ax = plt.subplots(figsize=(11, 9))

    # Fondo cartográfico de España
    espana.plot(
        ax=ax,
        color="#e8efe3",
        edgecolor="#52665b",
        linewidth=1.3
    )

    # Delimitar la España peninsular
    ax.set_xlim(-10, 4.5)
    ax.set_ylim(35.5, 44.5)
    ax.set_aspect("equal")

    # Dibujar las capitales
    ax.scatter(
        ubicaciones[:, 1],
        ubicaciones[:, 0],
        color="#173d68",
        s=22,
        zorder=5,
        label="Capitales"
    )

    # Crear la línea que cambiará en cada fotograma
    linea, = ax.plot(
        [],
        [],
        color="red",
        linewidth=1.6,
        alpha=0.85,
        zorder=4,
        label="Mejor recorrido"
    )

    # Texto con información de la iteración
    info = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=11,
        bbox=dict(
            facecolor="white",
            alpha=0.90,
            edgecolor="gray",
            boxstyle="round,pad=0.6"
        )
    )

    ax.set_title(
        "Optimización de rutas - Colonia de Hormigas",
        fontsize=15
    )

    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.15)

    # Función que actualiza cada fotograma
    def actualizar(frame):

        indice = indices[frame]

        ruta = rutas[indice]
        costo = costos[indice]

        # Obtener las coordenadas en orden de visita
        coordenadas = ubicaciones[ruta]

        linea.set_data(
            coordenadas[:, 1],
            coordenadas[:, 0]
        )

        info.set_text(
            f"Iteración: {indice + 1} / {len(rutas)}\n"
            f"Mejor costo: {costo:,.2f} euros"
        )

        if indice == (len(rutas)//intervalo)*intervalo+len(rutas)%intervalo-1:
                direccion_png = "outputs/Recorrido_ACO_" + version+".png"
                plt.savefig(direccion_png)

        return linea, info

    animacion = FuncAnimation(
        fig,
        actualizar,
        frames=len(indices),
        interval=1000 / fps,
        blit=False,
        repeat=False
    )

    direccion_gif = "outputs/Evolucion_ACO_"+version+".gif"

    # Exportación a GIF
    if exportar_gif:
        animacion.save(
            direccion_gif,
            writer=PillowWriter(fps=fps),
            dpi=90
        )
        print("GIF guardado correctamente.")

    # Exportación a MP4
    if exportar_mp4:
        writer = FFMpegWriter(
            fps=fps,
            codec="libx264",
            bitrate=1800,
            extra_args=["-pix_fmt", "yuv420p"]
        )

        direccion_mp4 = "outputs/Evolucion_ACO_"+version+".mp4"

        animacion.save(
            direccion_mp4,
            writer=writer,
            dpi=120
        )

        print("MP4 guardado correctamente.")

    plt.close(fig)

def main():
    mpl.rcParams["animation.ffmpeg_path"] = (
        imageio_ffmpeg.get_ffmpeg_exe()
    )

    ap = argparse.ArgumentParser()

    ap.add_argument("--valor-hora", type=float, required=True, help="EUR por hora del vendedor")
    ap.add_argument("--consumo", type=float, required=True, help="L/100 km (WLTP del vehículo elegido)")
    ap.add_argument("--precio", type=float, required=True, help="EUR por litro del combustible")
    ap.add_argument("--version", type=str, required=False)

    a = ap.parse_args()

    nombres, ubicaciones = leer_capitales()
    P = leer_peajes(len(nombres))

    problem, _ = matriz_costos(
    a.valor_hora, a.consumo, a.precio, P
)

    optimizer = AntColonyOptimizer(
    ants=100,
    evaporation_rate=0.05,
    intensification=2,
    alpha=2,
    beta=1,
    beta_evaporation_rate=0,
    choose_best=0.1
    )

    os.makedirs("outputs", exist_ok=True)

#Ejecutar optimización

    best = optimizer.fit(
    problem,
    iterations=500
    )

    print(f"Mejor costo encontrado: {best:.2f} euros")



#Cargar mapa de España

    URL_MAPA = (
        "https://raw.githubusercontent.com/"
        "nvkelso/natural-earth-vector/master/"
        "geojson/ne_110m_admin_0_countries.geojson"
    )

    mundo = gpd.read_file(URL_MAPA)
    espana = mundo[mundo["ADMIN"] == "Spain"]

    crear_animacion(
        optimizer,
        ubicaciones,
        espana,
        a.version,
        intervalo=5,
        fps=5
    )

if __name__ == "__main__":
    main()    