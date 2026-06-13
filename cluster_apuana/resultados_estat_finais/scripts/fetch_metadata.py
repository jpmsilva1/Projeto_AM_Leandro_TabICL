import openml
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

DATASETS = [
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
]

meta = []
for ds in DATASETS:
    tid = ds['tid']
    print(f"Buscando metadata para: {ds['name']} (tid: {tid})")
    try:
        task = openml.tasks.get_task(tid)
        did = task.dataset_id
        dataset = openml.datasets.get_dataset(did, download_data=False, download_qualities=True, download_features_meta_data=True)
    except:
        dataset = openml.datasets.get_dataset(tid, download_data=False, download_qualities=True, download_features_meta_data=True)
    
    quals = dataset.qualities
    n_classes = quals.get('NumberOfClasses', 2) # fallback to 2 if not found
    
    # check n_missing
    n_missing = quals.get('NumberOfMissingValues', 0)
    has_missing = n_missing > 0
    
    # features types
    features = dataset.features
    cat_feats = 0
    num_feats = 0
    for fid, f in features.items():
        if f.name == dataset.default_target_attribute:
            continue
        if f.data_type == 'nominal' or f.data_type == 'string':
            cat_feats += 1
        else:
            num_feats += 1
            
    total_feats = cat_feats + num_feats
    cat_ratio = cat_feats / total_feats if total_feats > 0 else 0
    
    meta.append({
        'dataset': ds['name'],
        'n_classes': int(n_classes),
        'has_missing': has_missing,
        'cat_ratio': cat_ratio,
        'is_binary': int(n_classes) == 2,
        'is_mostly_categorical': cat_ratio > 0.5
    })

pd.DataFrame(meta).to_csv('cluster_apuana/resultados_estat_finais/dataset_metadata.csv', index=False)
print("Metadata salvo em 'cluster_apuana/resultados_estat_finais/dataset_metadata.csv'")
