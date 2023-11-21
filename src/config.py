class Config(object):
    def __init__(self,
                 data_dir,
                 train_annotations,
                 val_annotations,
                 test_annotations,
                 sample_rate,
                 target_length,
                 batch_size,
                 shuffle_train,
                 shuffle_val,
                 shuffle_test,
                 apply_transforms,
                 mel_spec_params,
                 model_class_name,
                 learning_rate,
                 epochs,
                 model_path):
        self.data_dir = data_dir
        self.train_annotations = train_annotations
        self.val_annotations = val_annotations
        self.test_annotations = test_annotations
        self.sample_rate = sample_rate
        self.target_length = target_length
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.shuffle_val = shuffle_val
        self.shuffle_test = shuffle_test
        self.apply_transforms = apply_transforms
        self.mel_spec_params = mel_spec_params if apply_transforms else None
        self.model_class_name = model_class_name
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model_path = model_path
