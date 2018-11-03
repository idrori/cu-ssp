We provide clean datasets for training and test.

There are no duplicates in our training set.
Our training and test sets are disjoint.

The cb6133 datasets and splits that were provided by Princeton were corrupt:
1. cb6133 train and test contain duplicates
2. cb6133 train and test are not disjoint.
3. cb6133filtered contains duplicates.

Fortunately we found out the mistake right away: https://github.com/idrori/cu-ssp/tree/master/testing

One of our participants informed Princeton and they posted an update on 10.28.18 on their site, acknowledging their mistake:
https://www.princeton.edu/~jzthree/datasets/ICML2014/dataset_readme.txt

All comparisons with cb6133 in a dozen of follow-up papers between 2014-2018 compared with a corrupt benchmark. We are 76.3% on cb6133 which is best compared with the prevous 74.8%.

Fortunately, cb513 is disjoint from cb6133filtered and a valid benchmark.
