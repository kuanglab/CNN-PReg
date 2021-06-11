#!/usr/bin/env python

import command
import training

def main(args=None):
    
    parser = command.parser()
    args = parser.parse_args(args)
    
    # Debug
    print("debug: " + str(args))

    if args.task == "gene_clustering":
        training.run_gene_clustering(args)
    else:
        training.run_pretraining(args)

if __name__ == '__main__':
    main()