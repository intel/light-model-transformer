import pandas as pd
import os
from bs4 import BeautifulSoup
import matplotlib

# gradient colormap from red to green
colors = ["darkred", "red", "lightsalmon", "white", "lightgreen", "green", "lime"]
values = [0, 0.10, 0.2, 0.5, 0.8, 0.9, 1]
rgmap = matplotlib.colors.LinearSegmentedColormap.from_list(name="red-white-green", colors=list(zip(values, colors)))

# returns rgb according to the value
def color(val, cmap, vmin, vmax):
    norm = matplotlib.colors.Normalize(vmin, vmax)
    rgb = cmap(norm(abs(val)))[:3]
    return matplotlib.colors.rgb2hex(rgb)

def compare(fname, cfolder, header, results):
    Nightly_suffix = "_Nightly"
    PR_suffix      = ""
    Diff_suffix    = " Diff"

    data_pr = pd.read_csv(fname, sep='\t').dropna().drop_duplicates()
    data_n  = pd.read_csv(cfolder + os.sep + fname, sep='\t').dropna().drop_duplicates()

    try:
        output = pd.merge(data_n, data_pr,
                        on=header,
                        how='inner',
                        suffixes=[Nightly_suffix, PR_suffix])

        # st = output.style

        no_ms = lambda x: str(x).rstrip(' samples/s')

        for header in results:
            pr = output[header + PR_suffix].map(no_ms).astype(float)
            ni = output[header + Nightly_suffix].map(no_ms).astype(float)
            output[header + Diff_suffix] = (pr / ni * 100).apply(lambda x: "{0:.2f}%".format(x))
            # st.background_gradient(cmap=rgmap, vmin=0.5, vmax=1.5,
            #                        subset=header + ' Diff')

        # remove no longer needed columns with nightly results
        output.drop(output.filter(regex=Nightly_suffix).columns, axis=1, inplace=True)

        # columns to mark
        cols = []
        for header in results:
            cols.append(output.columns.get_loc(header + Diff_suffix))

        res = BeautifulSoup(output.to_html(index=False), features='html.parser')

        # set color of added columns according to it's values
        for tr in res.table.tbody.select('tr'):
            for i in cols:
                td = tr.select('td')[i]
                td.attrs['bgcolor'] = color(float(td.text.strip('%')) / 100.0,
                                            rgmap,
                                            0.5, 1.5)
    except KeyError:
        print("Something went wrong with merging, saving without comparison data")
        res = BeautifulSoup(data_pr.to_html(index=False), features='html.parser')

    # for some reason plugin ignores th* flags 🤨
    for th in res.table.thead.tr.select('th'):
        th.name = 'td'

    with open(os.path.splitext(fname)[0] + '.html','w') as f:
        f.write(str(res))

    # with open(os.path.splitext(fname)[0] + '.html','w') as f:
    #     f.write(str(st.to_html()))


if __name__ == '__main__':
    nightly_dir = "nightly"

    compare(fname   = 'accuracy.csv',
            cfolder = nightly_dir,
            header  = ['Compiler', 'Model', 'TF', 'Quantization', 'BFloat16'],
            results = ['Result'])
    compare(fname   = 'benchmark.csv',
            cfolder = nightly_dir,
            header  = ['Compiler', 'App', 'TF', 'Quantization', 'BFloat16', 'Batch Size'],
            results = ['Throughput'])
    compare(fname   = 'model_zoo.csv',
            cfolder = nightly_dir,
            header  = ['Compiler', 'Model', 'TF', 'Quantization', 'BFloat16', 'Batch Size'],
            results = ['Result', 'Throughput'])
