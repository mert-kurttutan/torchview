# Docs for Torchview

This is the docs folder for torchview project. This essentially builds static website using markdown files via [Docusaurus 2](https://docusaurus.io/)

## Structure

```bash
docs
├── README.md
├── babel.config.js
├── docs
│   ├── intro.md
│   ├── rolling_graph.md
│   ├── torch_function_notes.md
│   └── tutorial
├── docusaurus.config.js
├── package-lock.json
├── package.json
├── sidebars.js
├── src
│   ├── components
│   ├── css
│   └── pages
└── static
    └── img
```

`docs` folder is the most essential part of the part. It stores the markdown files used for the content of each webpage.

`docs` folder consists of 2 main parts. `tutorial` and others (to be changed later). `tutorial` part contains examples of how to use, and notebooks.
### Installation

```
$ yarn
```

### Local Development

```
$ yarn start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

### Build

```
$ yarn build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.


### Acknowledgement

Diagram part for the icon is taken from [here](https://www.svgrepo.com/svg/247576/diagram-order)

Flame part for the icon is taken from [here](https://www.svgrepo.com/)

The tool to convert svg to icon is the online web app, [here](https://redketchup.io/icon-converter). This website can give transparent background, which is not available in many other websites for converting to ico

The website used for editing and merging svg files, https://mediamodifier.com/svg-editor#