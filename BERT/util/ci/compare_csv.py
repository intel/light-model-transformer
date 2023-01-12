from cmath import nan
import pandas as pd
import os
from bs4 import BeautifulSoup
import matplotlib
from math import isnan

default_values = {
    'BERT variant': 'BERT-base'
}

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
    try:
        data_n  = pd.read_csv(cfolder + os.sep + fname, sep='\t').dropna().drop_duplicates()
    # If there's not nightly run to compare to, just save the .html
    except FileNotFoundError:
        print(f'No nightly data found for {fname}, saving without comparison.')
        res = BeautifulSoup(data_pr.to_html(index=False), features='html.parser')
        for th in res.table.thead.tr.select('th'):
            th.name = 'td'
        with open(os.path.splitext(fname)[0] + '.html','w') as f:
            f.write(str(res))
        return

    # If we add a new column to the table, we need to reindex the nightly DataFrame so that it has the same header.
    # The default np.NaN content of the new column may not make sense for the pd.merge call, so we can add an entry to 
    # the `default_values` dictionary for that particular column and fill the missing values. 
    data_n = data_n.reindex(columns=data_pr.columns).fillna(default_values)

    try:
        # Use 'outer' to include new rows, which were not previously run in nightly
        output = pd.merge(data_n, data_pr,
                        on=header,
                        how='outer',
                        suffixes=[Nightly_suffix, PR_suffix])

        # Drop NaNs except those in the nightly columns (these are expected due to 'outer' above).
        # This resutls in new rows being shown, and leaves 'nan%' in columns that show comparison against nightly.
        output.dropna(inplace=True, subset=[col for col in data_n.columns if Nightly_suffix not in col])

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
                try:
                    as_float = float(td.text.strip('%'))
                    value = as_float / 100.0
                except ValueError:
                    value = float('nan')

                if not isnan(value):
                    field_color = color(value, rgmap, 0.5, 1.5)
                else:
                    field_color = matplotlib.colors.rgb2hex((.5, .5, .5))
                td.attrs['bgcolor'] = field_color

    except KeyError:
        print("Something went wrong with merging, saving without comparison data")
        res = BeautifulSoup(data_pr.to_html(index=False), features='html.parser')

    # for some reason plugin ignores th* flags 🤨
    for th in res.table.thead.tr.select('th'):
        th.name = 'td'

    res.table.attrs['sorttable'] = "yes"

    # looks like sometimes plugin cannot handle thead and tbody tags
    res.table.thead.unwrap()
    res.table.tbody.unwrap()

    with open(os.path.splitext(fname)[0] + '.html','w') as f:
        f.write(str(res))

    return res, os.path.splitext(fname)[0]

    # with open(os.path.splitext(fname)[0] + '.html','w') as f:
    #     f.write(str(st.to_html()))

# Plugin correctly shows several tables on a page only when all of them are inside of one section
def summary(reports):
    res = BeautifulSoup()
    res.append(res.new_tag("tabs"))

    for report, caption in reports:
        tab = res.new_tag("tab")
        tab.attrs['name'] = caption
        tab.append(report)
        res.tabs.append(tab)

    with open('fullReport.html','w') as f:
        f.write(str(res))


if __name__ == '__main__':
    nightly_dir = "nightly"

    accuracy = compare(fname   = 'accuracy.csv',
            cfolder = nightly_dir,
            header  = ['Compiler', 'Model', 'TF', 'Quantization', 'BFloat16'],
            results = ['Result'])
    accuracy_pytorch = compare(fname   = 'accuracy_pytorch.csv',
            cfolder = nightly_dir,
            header  = ['Model', 'Quantization', 'BFloat16',],
            results = ['Accuracy'])
    benchmark = compare(fname   = 'benchmark.csv',
            cfolder = nightly_dir,
            header  = ['Compiler', 'App', 'TF', 'BERT variant', 'Quantization', 'BFloat16', 'Batch Size'],
            results = ['Throughput'])
    model_zoo = compare(fname   = 'model_zoo.csv',
            cfolder = nightly_dir,
            header  = ['Compiler', 'Model', 'TF', 'BERT variant', 'Quantization', 'BFloat16', 'Batch Size'],
            results = ['Result', 'Throughput'])

    summary([accuracy,
             accuracy_pytorch,
             benchmark,
             model_zoo])
