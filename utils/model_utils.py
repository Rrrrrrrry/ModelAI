# model_utils.py
import joblib
import pickle


def save_model(model, file_path, data_format='joblib'):
    """
    Save model to file_path
    :param model: model object
    :param file_path:
    :param data_format: data format ('joblib' or 'pickle')
    :return:
    """
    try:
        if data_format == 'joblib':
            joblib.dump(model, file_path)
        elif data_format == 'pickle':
            with open(file_path, 'wb') as f:
                pickle.dump(model, f)
        else:
            print(f"Unsupported data format: {data_format}")
            return

        print(f"Model saved successfully at {file_path}")
    except Exception as e:
        print(f"Error saving model: {e}")


def load_model(file_path, data_format='joblib'):
    """
    Load model from file_path
    :param file_path: file path to load
    :param data_format: data format ('joblib' or 'pickle')
    :return:
    """
    try:
        if data_format == 'joblib':
            model = joblib.load(file_path)
        elif data_format == 'pickle':
            with open(file_path, 'rb') as f:
                model = pickle.load(f)
        else:
            print(f"Unsupported data format: {data_format}")
            return None
        print(f"Model loaded successfully from {file_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
