import numpy
import os
import pandas
import matplotlib.pyplot as plot

MARKERS = [ "s", "o" ]
EDGES = [ "red", "none" ]
FACES = [ "none", "blue" ]

def run():
    simdir = "./run/sims/"
    for sim in os.listdir(simdir):
        
        path = simdir + sim
        square = path + "/square.csv"
        circle = path + "/circle.csv"
        if os.path.isfile(square):
            plotpath = f"{simdir}/{sim}/plots"
            if os.path.exists(plotpath):
                for filename in os.listdir(plotpath):
                    os.remove(f"{plotpath}/{filename}")
            else:
                os.makedirs(plotpath, exist_ok=True)
            data = [pandas.read_csv(square), pandas.read_csv(circle)]
            select = data[0].select_dtypes(include = ["number"]).columns.tolist()
            select = [column for column in data[0] if data[0][column].nunique() > 1 or data[1][column].nunique() > 1]
            leng = len(select)

            for x in range(leng):
                for y in range(leng):
                    if x == y:
                        continue
                    varX = select[x]
                    varY = select[y]
                    fig, ax = plot.subplots()
                    plot.title(f"{varX} vs {varY}")
                    plot.xlabel(f"{varX}\n\n")
                    for id, label in enumerate(["Square Grid", "Circle Grid"]):
                        if id:
                            ax = ax.twinx()

                        ax.set_ylabel(f"{varY}\n({label})")
                        dataX = data[id][varX]
                        dataY = data[id][varY]
                        ax.scatter(dataX, dataY, marker = MARKERS[id], label = label, facecolor = FACES[id], edgecolor = EDGES[id])
                        # coefficients = numpy.polyfit(dataX, dataY, 1)
                        # polynomial = numpy.poly1d(coefficients)
                        # xFit = numpy.linspace(dataX.min(), dataY.min(), 100)
                        # yFit = polynomial(xFit)
                        # plot.plot(xFit, yFit)
                    plot.grid(True)
                    fig.legend(loc="lower center", ncol=2)
                    plot.tight_layout()
                    plot.savefig(f"{simdir}/{sim}/plots/{varX} vs {varY}.png", format = "png")
                    plot.close()


run()
