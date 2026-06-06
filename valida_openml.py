import openml

datasets_to_test = [
    # Falharam (Bug do ID)
    {'tid': 3892, 'name': 'hiva_agnostic'},
    {'tid': 3481, 'name': 'isolet'},
    {'tid': 3510, 'name': 'JapaneseVowels'},
    {'tid': 3688, 'name': 'houses'},
    {'tid': 3945, 'name': 'KDDCup09_appetency'},
    
    # Pendentes (Não rodaram ainda)
    {'tid': 34539, 'name': 'Amazon_employee_access'},
    {'tid': 168868, 'name': 'APSFailure'},
    {'tid': 360945, 'name': 'KDD98'},
    {'tid': 361099, 'name': 'Diabetes130US'},
    {'tid': 359992, 'name': 'porto-seguro'}
]

print("Iniciando Validação de Qualidade de Dados (OpenML)")
print("="*60)

for ds in datasets_to_test:
    name = ds['name']
    tid = ds['tid']
    try:
        task = openml.tasks.get_task(tid)
        did = task.dataset_id
        
        # Pega APENAS OS METADADOS (não faz o download pesado dos dados)
        dataset = openml.datasets.get_dataset(did, download_data=False, download_qualities=False, download_features_meta_data=True)
        
        features = len(dataset.features)
        
        # Tenta pegar as qualidades (amostras e classes)
        # O OpenML pode não ter qualities se não baixar o dataset inteiro, mas meta_data ajuda
        try:
            qualities = dataset.qualities
            samples = int(qualities.get('NumberOfInstances', -1))
            classes = int(qualities.get('NumberOfClasses', -1))
        except Exception:
            samples = "Desconhecido"
            classes = "Desconhecido"
            
        print(f"✅ {name:25} (Task: {tid} | Real Dataset ID: {did}):")
        print(f"      - Features Totais: {features}")
        
    except Exception as e:
        print(f"❌ {name:25} Erro de Acesso OpenML: {e}")
