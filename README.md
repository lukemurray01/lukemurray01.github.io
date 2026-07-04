# lukemurray01.github.io

Personal portfolio, computational-physics showcase and blog for **Luke Murray** —
a physics & mathematics student focused on stochastic numerics (CIR & CEV) and a
quantitative, computational approach aimed at quant / data science / risk roles.

Live site: https://lukemurray01.github.io/

Built with **Jekyll** (native GitHub Pages) and a self-contained minimalist
theme — light, monochrome, typography-forward. No CSS framework.

## Structure

- `index.html` — landing page: hero, about, skills, featured work, latest posts, contact.
- `cv.html` — web CV / résumé (print-friendly; the "Download PDF" button prints to PDF,
  or drop `assets/Luke-Murray-CV.pdf` and point the link at it).
- `notes.html` — lecture-notes & solutions archive.
- `blog.html` + `_posts/` — blog (Markdown posts; MathJax for equations).
- Course pages — `Fouriermenu.html`, `Physicsmenu.html`, `Physics1.2menu.html`,
  `Classical/ClassicalMech.html`.
- Lecture notebooks (rendered via nbviewer) under `PY1053/`, `Fourier/`, `Electro/`, `Classical/`.
- `_layouts/`, `_includes/` — shared layout, nav, footer, `<head>`.
- `assets/css/site.css` — the entire theme.

## Add a blog post

Create `_posts/YYYY-MM-DD-slug.md` with front matter:

```yaml
---
layout: post
title: "Post title"
subtitle: "One-line summary"
date: 2026-08-01
tags: [Quant, Python]
mathjax: true        # only if the post uses LaTeX
reading_time: "8 min read"
---
```

It appears automatically on the blog index and the homepage feed.

## Local preview

```
bundle exec jekyll serve      # or: jekyll serve
```

## Editing tips

- **Headshot:** add `images/profile.jpg` and swap the `.avatar-ph` div in the hero of
  `index.html` for the commented-out `<img class="avatar" …>`.
- **LinkedIn:** replace `href="#"` on the LinkedIn link in `_includes/footer.html`.
- **CV PDF:** add `assets/Luke-Murray-CV.pdf` and point the CV button at it.

## Credits

Layout originally bootstrapped from Phantom by HTML5 UP; now a custom minimalist theme.
