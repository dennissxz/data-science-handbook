# MyST Markdown

ref: [docu](https://jupyterbook.org/content/myst.html)

## Math

No empty line in `align` environment, to avoid parse error.


## Special Content Block

admonition, attention, caution, danger, error, important, hint, note, seealso, tip and warning.


```{admonition} My title
:class: note
My content in {note} format
```

```{attention}
attention
```

```{caution}
caution
```

```{warning}
warning
```

```{danger}
danger
```

```{error}
error
```

```{important}
important
```

```{hint}
hint
```

```{note}
:class: dropdown
note + dropdown
```

```{seealso}
seealso
```

```{tip}
tip
```


Term 1
: Definition

Term 2
: Definition

## Hiding

[docu](https://jupyterbook.org/interactive/hiding.html?highlight=hide)

- add `:tags: [hide-input]` to code cell

## Coloring

`<span style="color:red">colored text</span>`: <span style="color:red">colored text</span>
