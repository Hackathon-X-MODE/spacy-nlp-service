from ml.model.model import REF, MAIN_TYPE_MODEL, PREPARE_ORDER_SUB_TYPE_MODEL

for model in REF:
    print("Runner... ", model)
    REF[model].traning()
