### how do we preprocess an image?
- read file into a byte string;
- decode jpeg;
- resize image;
- if this is train image - flip it;
- cast to float32;
- preprocess to xception; 

### how do we prepare a dataset?
- map the function with above operations;
- if train repeat and shuffle;
- batch and prefetch;