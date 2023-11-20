import argparse

def PROPOSED_PARSER():
    parser = argparse.ArgumentParser(description=None)
    
    parser.add_argument('--cuda', type=int, default=0)

    parser.add_argument('--n-fft', type=int, default=1193)
    parser.add_argument('--hop-length', type=int, default=9)
    parser.add_argument('--freq-range', type=int, default=224)
    parser.add_argument('--num-output', type=int, default=11)
    
    parser.add_argument('--labeled-data-N', type=int, default=1)
    parser.add_argument('--labeled-train-amount', type=int, default=None, choices=[None, 100, 200])

    parser.add_argument('--x-scaling', type=str, default='minmax')
    parser.add_argument('--y-scaling', type=str, default="minmax",choices=['raw', 'minmax'])
    
    parser.add_argument('--epochs', type=int, default=50,help='number of epochs to train (default: auto)')
    parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training (default: auto)')
    parser.add_argument('--lr', type=float, default=1e-3,help='learning rate (default: auto)')
    
    parser.add_argument('--masking-range', type=int, default=20)    

    parser.add_argument('--lambda-u', type=float, default=0.05) 
    parser.add_argument('--warm-up', type=float, default=20)
    parser.add_argument('--seed', type=int, default=2022)

    return parser