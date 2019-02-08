# WiDS2019

## File descriptions

- train_images/image_[9-digit number].jpg - the training images from Planet
- traininglabels.csv - the crowdsourced annotations/labels of presence or absence of oil palm in the training data, from Figure Eight
- leaderboard_test_data/image_[9-digit number].jpg - the test images from Planet
- SampleSubmission.csv - a sample submission file in the correct format


## Data fields
- image_id - an anonymous id unique to a given image
- has_oilpalm - the annotation or label for a given image, with 0 indicating no oil palm, and 1 indicating presence of oil palm plantations
- score - confidence score based on the aggregated results from crowdsourcing the annotations. This describes the level of agreement between multiple contributors, weighted by the contributor's trust score, and indicates Figure Eight's confidence in the validity of the result. For more details on how these scores are calculated, visit this article. Please note that this is extra data that need not be incorporated in your model, but may be useful. In addition, stay tuned for a blogpost focused more deeply on the data annotation process and more!
