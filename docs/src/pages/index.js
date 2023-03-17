import React from 'react';
import clsx from 'clsx';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';

import styles from './index.module.css';

import ThemedImage from '@theme/ThemedImage';
import useBaseUrl from '@docusaurus/useBaseUrl';



function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <h1 className="hero__title">{siteConfig.title}</h1>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <p align="center">
        <ThemedImage
          alt="Docusaurus themed image"
          sources={{
            light: useBaseUrl('https://user-images.githubusercontent.com/88637659/213171745-7acf07df-6578-4a50-a106-1a7b368f8d6c.svg'),
            dark: useBaseUrl('https://user-images.githubusercontent.com/88637659/213173736-6e91724c-8de1-4568-9d52-297b4b5ff0d2.svg'),
          }}
        />;
        </p>
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title}`}
      description="Description will go into a meta tag in <head />">
      <HomepageHeader />
      <main>
        {/* <HomepageFeatures /> */}
      </main>
    </Layout>
  );
}
