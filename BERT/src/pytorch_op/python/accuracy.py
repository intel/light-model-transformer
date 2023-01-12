#!/usr/bin/env python
import torch

import numpy as np
import pandas as pd

import transformers

from datasets import load_dataset

import itertools
import argparse
import time

QUANT_FACTORS=[
        -10.85244083404541015625, 4.14164829254150390625, -1.6212508678436279296875, 2.18305110931396484375, -64.5349578857421875, 9.17784881591796875, -0.16926576197147369384765625, 12.69039154052734375,
        -10.01922702789306640625, 3.2598330974578857421875, -2.52011966705322265625, 3.17220592498779296875, -70.322662353515625, 4.564808368682861328125, -0.16925294697284698486328125, 10.93472957611083984375,
        -11.37454319000244140625, 4.04611110687255859375, -2.5044767856597900390625, 3.4310567378997802734375, -56.21540069580078125, 5.208764553070068359375, -0.16948534548282623291015625, 72.20577239990234375,
        -14.79791736602783203125, 4.259090423583984375, -2.8403589725494384765625, 3.91925144195556640625, -93.42563629150390625, 5.099577426910400390625, -0.1689991652965545654296875, 9.5706195831298828125,
        -13.21285343170166015625, 4.449753284454345703125, -3.1772515773773193359375, 4.3330135345458984375, -101.334869384765625, 5.41256046295166015625, -0.16838109493255615234375, 10.64498996734619140625,
        -13.93945217132568359375, 5.1448192596435546875, -2.5481836795806884765625, 3.48368167877197265625, -91.05278778076171875, 5.9057769775390625, -0.16948328912258148193359375, 12.6811923980712890625,
        -14.12649059295654296875, 5.23845577239990234375, -2.814735889434814453125, 3.2215893268585205078125, -89.623870849609375, 6.68107700347900390625, -0.16898013651371002197265625, 11.01731777191162109375,
        -13.5746974945068359375, 4.71494960784912109375, -2.7004568576812744140625, 3.2631299495697021484375, -87.90279388427734375, 7.388260364532470703125, -0.16951541602611541748046875, 8.03197765350341796875,
        -15.597011566162109375, 6.920653820037841796875, -3.0222375392913818359375, 3.777666568756103515625, -83.6142730712890625, 10.2494525909423828125, -0.1686449944972991943359375, 23.9402790069580078125,
        -15.88373565673828125, 10.81757640838623046875, -2.6777179241180419921875, 3.3885133266448974609375, -48.061458587646484375, 16.7345333099365234375, -0.156786620616912841796875, 92.52396392822265625,
        -18.6183719635009765625, 11.54715251922607421875, -2.11896610260009765625, 3.066336154937744140625, -41.8497314453125, 19.4496479034423828125, -0.16698478162288665771484375, 141.4157867431640625,
        -23.8061676025390625, 11.55181217193603515625, -2.552584171295166015625, 3.7034885883331298828125, -36.45532989501953125, 16.997623443603515625, -0.16963402926921844482421875, 8.112117767333984375
    ]

def run_model(model, config, inputs, labels, results):
    model = transformers.BertForSequenceClassification.from_pretrained(model, config=config)
    model.eval()

    with torch.no_grad():
        output = model(**inputs).logits
        output = np.argmax(output.numpy(), -1)

    correct = np.sum(labels == output)
    num_samples = len(labels)
    accuracy = correct / num_samples

    return correct, num_samples, accuracy
    

def main(args: argparse.Namespace):
    dataset = load_dataset('glue', 'mrpc')
    data = dataset['test']

    samples = data[:args.num_samples]
    labels = np.array(samples['label'])

    tokenizer = transformers.BertTokenizer.from_pretrained(args.tokenizer)
    inputs = tokenizer(text=samples['sentence1'], text_pair=samples['sentence2'], return_tensors='pt', max_length=512, padding='max_length')

    results = pd.DataFrame(columns=['Model', 'Quantization', 'BFloat16', 'Correct/Total', 'Accuracy'])


    # Unoptimized model run
    print('Running unoptimized model')
    config = transformers.BertConfig.from_pretrained(args.model)
    correct, num_samples, accuracy = run_model(args.model, config, inputs, labels, results)
    results = results.append({
        'Model': args.model,
        'Quantization': '-',
        'BFloat16': '-',
        'Correct/Total': f'{correct}/{num_samples}',
        'Accuracy': f'{accuracy}'
        },
        ignore_index=True)

    # Optimized model runs:
    import bert_op

    quant_values = [True, False]
    bf_values = [False, True]

    config = transformers.BertConfig.from_pretrained(args.model)
    for quant, bf16 in itertools.product(quant_values, bf_values):
        # Skip Quant+BF16 due to very slow implementations for pre-SPR HW
        if quant and bf16:
            continue

        print(f'Running optimized model: Quantization = {quant}, BFloat16 = {bf16}')
        config.use_quantization = quant
        config.use_bfloat16 = bf16
        config.quantization_factors = QUANT_FACTORS
        correct, num_samples, accuracy = run_model(args.model, config, inputs, labels, results)
        results = results.append({
            'Model': '(OPT) ' + args.model,
            'Quantization': 'On' if quant else 'Off',
            'BFloat16': 'On' if bf16 else 'Off',
            'Correct/Total': f'{correct}/{num_samples}',
            'Accuracy': f'{accuracy}'
            },
            ignore_index=True)

    print(results)

    if args.csv is not None:
        results.to_csv(args.csv, sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='An MRPC accuracy script for HuggingFace BERT models optimized with the BertOp.')

    parser.add_argument('-m', '--model', type=str, default='Intel/bert-base-uncased-mrpc')
    parser.add_argument('-t', '--tokenizer', type=str, default='bert-base-uncased', help='Tokenizer to used for the model.')

    parser.add_argument('-n', '--num_samples', type=int, default='100', help='Number of samples to run.')

    parser.add_argument('--output-csv', dest='csv', type=str, default=None,
                        help='If provided, results will be saved to a .csv file at this location.')
    parser.add_argument('--sep', default='\t', help='Separator used when writing to the .csv file.')


    args = parser.parse_args()
    main(args)
