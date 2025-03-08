import os
import pickle

def train(
    controller, 
    train_inputs, train_labels, 
    concept, model_name, path
):
    controller.compute_directions(train_inputs, train_labels)
    controller.save(concept, model_name, path)
    return controller.directions, controller.signs, controller.detector_coefs

def eval(
    controller, 
    valid_inputs, valid_labels, 
    test_inputs, test_labels, 
    concept, model_name, path
):
    controller.load(concept, model_name, path)
    valid_metrics, \
    test_metrics, \
    detector_coefs = controller.evaluate_directions(
        valid_inputs, valid_labels,
        test_inputs, test_labels,
    )
    control_method = controller.control_method
    save_metric(control_method, concept, model_name, path, valid_metrics, name="valid-metrics")
    save_metric(control_method, concept, model_name, path, test_metrics, name="test-metrics")
    save_metric(control_method, concept, model_name, path, detector_coefs, name="detector-agg")
    return valid_metrics, test_metrics, detector_coefs

def save_metrics(control_method, concept, model_name, path, data, name):
    filename = os.path.join(path, f'{control_method}_{concept}_{model_name}_{name}.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(data, f)