UNKNOWN_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"  # Optional: this is used to pad a batch of sentences in different lengths.
# ROOT_TOKEN = PAD_TOKEN # this can be used if you are not padding your batches
# ROOT_TOKEN = "<root>" # use this if you are padding your batches and want a special token for ROOT
SPECIAL_TOKENS = [PAD_TOKEN, UNKNOWN_TOKEN]