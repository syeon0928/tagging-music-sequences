import importlib

# List of model module names (without .py extension)
model_modules = ['model_ab', 'model_sy', 'model_yc', 'model_att_yc']

# Dynamically import all models
for module_name in model_modules:
    module = importlib.import_module(f'src.{module_name}')
    globals().update({k: v for k, v in module.__dict__.items() if not k.startswith('_')})