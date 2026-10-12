# Publishing the student worksheets

The website is built from this student repository. Quarto renders `index.qmd`
and every `exercises/**/worksheet.qmd` into `_site/`. Python exercises are run by
students locally; publishing the static worksheets needs no Python packages or
Jupyter kernel.

## Enable GitHub Pages once

1. Open [the repository's Pages settings](https://github.com/HealthML/deep-learning-exercises/settings/pages).
2. Under **Build and deployment → Source**, choose **GitHub Actions**. The
   repository already supplies its own workflow, so skip the suggested templates.
3. Merge the publishing pull request into `main`.
4. In [Actions](https://github.com/HealthML/deep-learning-exercises/actions/workflows/publish.yml),
   wait for **Publish student worksheets** to finish. The successful deployment
   links to the live site.

If the first deployment ran before Pages was enabled, enable it and use
**Re-run failed jobs** on that run. You can also select **Run workflow** with
the `main` branch. No personal access token or repository secret is needed.

## Add the link in Moodle once

With course editing enabled, choose **Add an activity or resource → URL**.
Use **Deep Learning classroom exercises** as the name and this External URL:

<https://healthml.github.io/deep-learning-exercises/>

For a link to just the Week 1 worksheet, use:

<https://healthml.github.io/deep-learning-exercises/exercises/week01/worksheet.html>

The previous `/exercises/week1/worksheet.html` address redirects to this page,
so existing Moodle links continue to work.

Choose **Open** or **New window** under Appearance so students can use the full
page. Later changes appear at the same address after a successful deployment.

## Update the material

Open a pull request with the source changes. The publishing workflow renders
the site on pull requests to catch rendering errors. Merging into `main` renders
and publishes the site automatically. The existing exercise checks still run
separately. A failed build does not replace the previously published site.

To add another worksheet, use a two-digit week folder, such as
`exercises/week02/worksheet.qmd` through `exercises/week15/worksheet.qmd`,
and add a link in `index.qmd`. The render list already includes that path pattern.

For a local preview, install Quarto and run these commands from the repository
root:

```sh
quarto preview
```

To render without starting a preview server:

```sh
quarto render --to html
```

Open `_site/index.html` or `_site/exercises/week01/worksheet.html`. The website
shares the black and light themes; equations use native MathML. Each HTML page
embeds its styles and images for offline reading. Keep generated `_site/`
output out of version control.
