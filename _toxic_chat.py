from datasets import load_dataset

def get_data(format_prompt):
    """https://huggingface.co/datasets/lmsys/toxic-chat"""
    # Load the dataset
    ds = load_dataset("lmsys/toxic-chat", "toxicchat1123")
    
    # Extract and preprocess inputs
    train_inputs = [x['user_input'] for x in ds['train']]
    test_inputs = [x['user_input'] for x in ds['test']]

    # Format prompts
    train_inputs = [format_prompt(x) for x in train_inputs]
    test_inputs = [format_prompt(x) for x in test_inputs]

    # Extract labels
    train_labels = [x['toxicity'] for x in ds['train']]
    test_labels = [x['toxicity'] for x in ds['test']]
    
    # Split into val/train
    n = len(train_inputs)
    
    return (
        train_inputs[:n//2], 
        train_labels[:n//2], 
        train_inputs[n//2:], 
        train_labels[n//2:], 
        test_inputs, 
        test_labels
    )

def get_toxic_splits(format_prompt):
    """Prompts formatted by either a pretrained or finetuned model tokenizer"""
    return get_data(format_prompt)

def get_instruction_splits(format_prompt):
    """Prompts formatted by a finetuned model tokenizer"""
    # Dataset without instruction prompts
    train_inputs, train_labels, \
    valid_inputs, valid_labels, \
    test_inputs, test_labels = get_data(format_prompt = lambda x: x)

    # Negative class: no instruction prompt
    train_labels = [0 for _ in train_labels]
    valid_labels = [0 for _ in valid_labels]
    test_labels = [0 for _ in test_labels]
    
    # Add instruction prompts
    train_inputs += [format_prompt(x) for x in train_inputs]
    valid_inputs += [format_prompt(x) for x in valinputs]
    test_inputs += [format_prompt(x) for x in test_inputs]

    # Positive class: instruction prompt
    train_labels += [1 for _ in train_labels]
    valid_labels += [1 for _ in valid_labels]
    test_labels += [1 for _ in test_labels]

    return (
        train_inputs, 
        train_labels, 
        valid_inputs, 
        valid_labels, 
        test_inputs, 
        test_labels
    )