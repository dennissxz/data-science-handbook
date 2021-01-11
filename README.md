# Data Science Handbook

This is a book that summarizes theories, models, and interview questions in data science.
Click [here](https://dennissxz.github.io/data-science-handbook/) to view the book online.

## Build the Book

The book is built by [Jupyter Book](https://jupyterbook.org/intro.html).

To make the code in `.md` Markdown files executable when the book is built, run the following command to convert the Markdown files to MyST Markdown files first. See [here](https://jupyterbook.org/file-types/myst-notebooks.html) for details.
```bash
jupyter-book myst init filename.md --kernel kernelname
```

Then, to build the book, run
```bash
cd ..
jupyter-book build data-science-handbook/
```
It will execute and merge all `.md` and `.ipynb` files according to the table of contents structure specified in `_toc.yml`, and output HTML files to `_build/html`.

To publish the local HTML files to Github pages, run the following command from the root of this repository. See [here](https://jupyterbook.org/publish/gh-pages.html#push-your-book-to-a-branch-hosted-by-github-pages) for details.
```bash
ghp-import -n -p -f _build/html
```
