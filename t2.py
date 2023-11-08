from transformers import AutoTokenizer, AutoModelForSequenceClassification

# instantiate tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')

# save the model and tokenizer
model.save_pretrained('my_model_directory')
tokenizer.save_pretrained('my_model_directory')

# load the model and tokenizer
loaded_tokenizer = AutoTokenizer.from_pretrained('my_model_directory')
loaded_model = AutoModelForSequenceClassification.from_pretrained('my_model_directory')


