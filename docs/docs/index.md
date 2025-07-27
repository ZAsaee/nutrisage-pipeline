# nutrisage-mlops documentation!

## Description

Healthy‑food intelligence, built with full‑stack MLOps on AWS

## Commands

The Makefile contains the central entry points for common tasks related to this project.

### Syncing data to cloud storage

* `make sync_data_up` will use `aws s3 sync` to recursively sync files in `data/` up to `s3://nutrisage/data/`.
* `make sync_data_down` will use `aws s3 sync` to recursively sync files from `s3://nutrisage/data/` to `data/`.


