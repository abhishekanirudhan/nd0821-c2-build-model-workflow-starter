#!/usr/bin/env python
"""
Download raw data from W&B, clean it and re-upload the clean dataset
"""
import argparse
import logging
import os

import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()
    
    logger.info("Downloading and reading input artifact as csv")

    local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(local_path)

    logger.info("Removing outliers in house price")
    min_price = args.min_price
    max_price = args.max_price
    idx = df["price"].between(min_price, max_price)
    df = df[idx].copy()

    logger.info("Converting last review from str to datetime")
    df['last_review'] = pd.to_datetime(df['last_review'])

    logger.info("Enforcing latitude and longitude bounds")
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    logger.info("Writing out cleaned dataset")
    
    df.to_csv(args.output_artifact, index = False)

    logger.info("Creating and logging wandb artifact for cleaned data")
    artifact = wandb.Artifact(
        args.output_artifact,
        type = args.output_type,
        description = args.output_description,
    )

    artifact.add_file(args.output_artifact)
    run.log_artifact(artifact)
    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="basic data cleaning step")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Name of the input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Name of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Type of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description of the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum price to remove outliers",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum price to remove outliers",
        required=True
    )


    args = parser.parse_args()

    go(args)
