class Config(object):
    def __init__(self,
                 data_dir,
                 train_annotations=None,
                 val_annotations=None,
                 test_annotations=None,
                 sample_rate=16000,
                 target_length=29.1,
                 batch_size=32,
                 num_workers=0,
                 apply_augmentations=False,
                 model_class_name=None,
                 learning_rate=None,
                 epochs=None,
                 model_path=None,
                 results_dir=None):
        self.data_dir = data_dir
        self.train_annotations = train_annotations
        self.val_annotations = val_annotations
        self.test_annotations = test_annotations
        self.sample_rate = sample_rate
        self.target_length = target_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.apply_augmentations = apply_augmentations
        self.model_class_name = model_class_name
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model_path = model_path
        self.results_dir = results_dir
