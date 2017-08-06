from matplotlib.lines import Line2D

def ohlc_chart(ax, quotes, width=0.2, colorup='r', colordown='k',linewidth=0.5):
    OFFSET = width / 2.0
    lines = []
    openlines = []
    closelines = []
    for q in quotes:
        t, open, high, low, close = q[:5]

        if close > open:
            color = colorup
        else:
            color = colordown

        vline = Line2D( xdata=(t, t), ydata=(low, high), color=color, linewidth=linewidth, antialiased=True)
        lines.append(vline)

        openline = Line2D(xdata=(t - OFFSET, t), ydata=(open,open), color=color, linewidth=linewidth, antialiased=True)
        openlines.append(openline)

        closeline = Line2D(xdata=(t , t+OFFSET), ydata=(close,close), color=color, linewidth=linewidth, antialiased=True)
        closelines.append(closeline)

        ax.add_line(vline)
        ax.add_line(openline)
        ax.add_line(closeline)
    
    ax.axhline(linewidth=1, color='g')
    ax.axvline(x=int(t/2), linewidth=1, color='g')
    ax.autoscale_view()

    return lines, openlines, closelines