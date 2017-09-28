import os
#import base64
#from Crypto.Cipher import XOR

#from matplotlib.lines import Line2D
from datetime import datetime, timedelta

import numpy as np
import scipy.sparse as sp

def get_tdop_array(density, info, now=None):
    """
    pytables density table로부터 정해진 날짜의 tdop array를 반환한다.
    args: 
     density: pytables group,
     tickunit: 틱단위,
     digit: 소숫점자릿수,
     name: 상품명
     code: 상품코드
     now: numpy datetime64 object 
    """

    if now == None:
        now = np.datetime64(datetime.now()+timedelta(hours=1)) #1시간 시차
        
    tickunit = info['tick_unit']
    digit = info['decimal_places']
    name = info['name']
    code = info['group']

    #print("processing: ", name)
    # data preprocessing
    dates = density.DateMapper.read(field='date').astype('M8[s]')
    source = density.Minute.read()
    price = source['price'].round(digit)
    row_num = source['row']
    col_num = np.rint((price-price.min())/tickunit)
    value = source['value']
    value[value == np.inf] = 0 #inf --> 0 
    value[value > value.std()*15] = 0 # larger than std * 15 --> 0
    value[value == np.NaN] = 0 # nan --> 0
    
    #sparse matrix creation
    shape = (row_num.max()+1, col_num.max()+1)
    matrix = sp.csr_matrix((value, (row_num, col_num)), shape=shape)
    
    #scale factor: sqrt(date - date)
    delta = (now - dates)/np.timedelta64(1,'D') # 시차(일수)
    delta[delta<0] = np.nan 
    delta = delta +1 # 최소시차 = 1
    seq = 1/np.sqrt(delta)
    seq[np.isnan(seq)] = 0
    scale = sp.diags(seq) #diagonal matrix
    
    #normalized TDOP
    tdop = np.squeeze(np.asarray((scale*matrix).sum(axis=0)))
    normed_tdop = tdop / tdop.sum()
    x_ticks = np.arange(price.min(), price.max()+tickunit/2, tickunit).round(digit)
    
    return x_ticks, normed_tdop, now


def norm(data, ntype='abs_diff'):
    """
    Data Normalization
    """
    if ntype=="abs_diff":
        """
        mean: 0
        scale factor: absolute diff mean
        """
        base = np.abs(data.diff()).mean()
        return (data-data.mean())/base

    if ntype=='min_max':
        """
        (data - min)/(max-min)
        """
        return (data-data.min())/(data.max()-data.min())

    if ntype=='zscore':
        return (data-data.mean())/data.std()




"""
캔들 차트
"""
def ohlc_chart(ax, quotes, linewidth=1, color='k'):
    dates = quotes.index.values
    ohlc = quotes[['open','high','low','close']].values
    o,h,l,c = np.squeeze(np.split(ohlc, 4, axis=1))
    offset = np.timedelta64(8, 'h')

    ax.vlines(dates, l, h, linewidth=linewidth, color=color)
    ax.hlines(o, dates-offset, dates, linewidth=linewidth, color=color)
    ax.hlines(c, dates, dates+offset, linewidth=linewidth, color=color)

    #style
    ax.grid(linestyle='--')
    ax.set_facecolor('lightgoldenrodyellow')
    ax.yaxis.tick_right()
    return ax

def ohlc_chart2(ax, quotes, width=0.2, colorup='r', colordown='k',linewidth=0.5):
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
    

    ax.autoscale_view()

    return lines, openlines, closelines


"""
 암호화 매소드
"""
def encrypt(key, plaintext):
  cipher = XOR.new(key)
  return base64.b64encode(cipher.encrypt(plaintext))

def decrypt(key, ciphertext):
  cipher = XOR.new(key)
  return cipher.decrypt(base64.b64decode(ciphertext))