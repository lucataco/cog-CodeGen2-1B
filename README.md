# Salesforce/codegen2-1B Cog model

This is an implementation of the [Salesforce/codegen2-1B](https://huggingface.co/Salesforce/codegen2-1B) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights

Then, you can run predictions:

    cog predict -i prompt="monkey scuba diving"
