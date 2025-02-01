from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def get_model(model_type='svm'):
    if model_type == 'svm':
        return SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    elif model_type == 'mlp':
        return MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42, learning_rate_init=0.001)
    else:
        raise ValueError("Invalid model type. Choose 'svm' or 'mlp'.")

