import openml

datasets = [
    {'tid': 32, 'name': 'pendigits', 'regime': 'large'},
    {'tid': 26, 'name': 'nursery', 'regime': 'large'},
    {'tid': 6, 'name': 'letter', 'regime': 'large'},
    {'tid': 3688, 'name': 'houses', 'regime': 'large'},
    {'tid': 359979, 'name': 'Amazon_employee_access', 'regime': 'large'},
    {'tid': 3945, 'name': 'KDDCup09_appetency', 'regime': 'large'},
    {'tid': 168868, 'name': 'APSFailure', 'regime': 'large'},
    {'tid': 361329, 'name': 'KDD98', 'regime': 'large'},
    {'tid': 211986, 'name': 'Diabetes130US', 'regime': 'large'},
    {'tid': 360113, 'name': 'porto-seguro', 'regime': 'large'},
    {'tid': 3892, 'name': 'hiva_agnostic', 'regime': 'medium'},
    {'tid': 43, 'name': 'spambase', 'regime': 'medium'},
    {'tid': 58, 'name': 'waveform-5000', 'regime': 'medium'},
    {'tid': 359968, 'name': 'churn', 'regime': 'medium'},
    {'tid': 30, 'name': 'page-blocks', 'regime': 'medium'},
    {'tid': 28, 'name': 'optdigits', 'regime': 'medium'},
    {'tid': 2074, 'name': 'satimage', 'regime': 'medium'},
    {'tid': 3481, 'name': 'isolet', 'regime': 'medium'},
    {'tid': 24, 'name': 'mushroom', 'regime': 'medium'},
    {'tid': 3510, 'name': 'JapaneseVowels', 'regime': 'medium'},
    {'tid': 1464, 'name': 'blood-transfusion-service-center', 'regime': 'small'},
    {'tid': 37, 'name': 'diabetes', 'regime': 'small'},
    {'tid': 2, 'name': 'anneal', 'regime': 'small'},
    {'tid': 168757, 'name': 'credit-g', 'regime': 'medium'},
    {'tid': 359956, 'name': 'qsar-biodeg', 'regime': 'medium'},
    {'tid': 2077, 'name': 'baseball', 'regime': 'medium'},
    {'tid': 2073, 'name': 'yeast', 'regime': 'medium'},
    {'tid': 45, 'name': 'splice', 'regime': 'medium'},
    {'tid': 359967, 'name': 'Bioresponse', 'regime': 'medium'},
    {'tid': 3011, 'name': 'hypothyroid', 'regime': 'medium'},
]

print("\\begin{table}[h!]")
print("\\centering")
print("\\small")
print("\\caption{Características dos 30 datasets avaliados no benchmark TabArena-v0.1.}")
print("\\label{tab:datasets}")
print("\\vspace{0.2cm}")
print("\\begin{tabular}{@{}llrrrc@{}}")
print("\\toprule")
print("\\textbf{Dataset} & \\textbf{TID (Task)} & \\textbf{Instâncias} & \\textbf{Features} & \\textbf{Classes} & \\textbf{Regime} \\\\ \\midrule")

data_rows = []
for d in datasets:
    try:
        try:
            task = openml.tasks.get_task(d['tid'])
            did = task.dataset_id
        except:
            did = d['tid']
        ds = openml.datasets.get_dataset(did, download_data=False, download_features_meta_data=False, download_qualities=True)
        quals = ds.qualities
        instances = int(quals.get('NumberOfInstances', 0))
        features = int(quals.get('NumberOfFeatures', 0))
        classes = int(quals.get('NumberOfClasses', 2))
        
        # fix missing classes if 0
        if classes == 0:
            classes = 2 # assume binary for these specific tasks

        data_rows.append({
            'name': d['name'].replace('_', '\\_'),
            'tid': d['tid'],
            'instances': instances,
            'features': features,
            'classes': classes,
            'regime': d['regime'].capitalize()
        })
    except Exception as e:
        data_rows.append({
            'name': d['name'].replace('_', '\\_'),
            'tid': d['tid'],
            'instances': '-',
            'features': '-',
            'classes': '-',
            'regime': d['regime'].capitalize()
        })

# Sort by instances
data_rows.sort(key=lambda x: x['instances'] if isinstance(x['instances'], int) else 0)

for r in data_rows:
    inst_str = f"{r['instances']:,}".replace(',', '.') if isinstance(r['instances'], int) else r['instances']
    print(f"{r['name']} & {r['tid']} & {inst_str} & {r['features']} & {r['classes']} & {r['regime']} \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")
