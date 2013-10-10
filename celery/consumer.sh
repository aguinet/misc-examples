#!/bin/bash
#

celery worker -A tasks:app -l info
