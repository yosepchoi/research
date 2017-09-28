import os
import h5py
import tables as tb


from ..config import DATADIR

def open_file(module, fname, mode='r', comp=False, force=False):
    """
    모듈별 file object를 생성
    """
    fpath = os.path.join(DATADIR, fname)
    
    if not force and mode=='w' and os.path.isfile(fpath):
        raise FileExistsError("file '%s' alreay exist"%fname)
    else:
        if module == 'h5py':
            return h5py.File(fpath, mode)

        elif module == 'tb':
            if comp:
                filters = tb.Filters(complib='blosc', complevel=9)
                return tb.open_file(fpath, mode=mode, filters=filters)
            else:
                return tb.open_file(fpath, mode=mode)


def product_info():
    import json

    fpath = os.path.join(DATADIR, 'product_info.json')
    fobj = open(fpath).read()
    return json.loads(fobj)


#def product_info():
#    """
#    DB에서 종목정보 불러와 Dict type으로 리턴
#    """
#    fpath = os.path.join(DATADIR, 'db.sqlite3')
#    con = lite.connect(fpath)
#    products = pd.read_sql('select * from trading_product', con)
#    products.set_index(['group'], drop=False, inplace=True)
#    products = products.to_dict(orient='index')
#    return products