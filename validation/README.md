# Independent check against the ORL experiment

Section 4.1 of [Li and Kuang's paper](https://jds-online.org/journal/JDS/article/1397/info) reports that the upper-edge and pairwise rules select three and four components when the first five images of each person serve as the training set. This script recreates that split and checks the ranks and the recognition counts seen in our own run. It computes the distances directly from feature differences. The clothing experiment in the parent directory uses Fashion-MNIST.

Run these commands from the repository root in a Python environment with NumPy and Pillow. We tested NumPy 2.3.5 and Pillow 13.0.0. The archive is from the AT&T Laboratories Cambridge collection maintained by the University of Cambridge. Its SHA-256 digest is `509d763bd2276aa054d260d7f1000e8be212738134f2f83d8ff13bc2adf1c8ec`.

```sh
python -m pip install numpy Pillow
curl -fL 'https://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip' -o /tmp/att_faces.zip
unzip -oq /tmp/att_faces.zip -d /tmp/att_faces
python validation/check_orl.py /tmp/att_faces
```

The same 400 source images are available in the [journal supplement](https://jds-online.org/journal/JDS/article/1397/file/13433). Extract the nested `code and data_JDS.zip` and give `check_orl.py` the path of the extracted `code and data/att_faces` directory. The authors' reference notebook is `code and data/codes/eigenvalue_choice_orl.ipynb` in that archive.

The script prints JSON and exits successfully if it matches `orl_expected.json`. With 100 sign trials and seed 2026, we obtained 180 correct out of 200 test images at rank 3 and 183 correct out of 200 at rank 4. Those are counts from the five-training-images split. The paper's table averages accuracies across five different splits.
