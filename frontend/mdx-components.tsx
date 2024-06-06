import type { MDXComponents } from 'mdx/types';
import classnames from 'classnames';
import classes from './mdx-components.module.css';

 
export function useMDXComponents(components: MDXComponents): MDXComponents {
  return {
    h1: ({ children }) => <h1 className={classes.h1}>{children}</h1>,
    h2: ({ children }) => <h2 className={classes.h2}>{children}</h2>,
    h3: ({ children }) => <h3 className={classes.h3}>{children}</h3>,
    p: ({ children }) => <p className={classes.p}>{children}</p>,
    a: ({ children, ...rest}) => <a className={classes.a} {...rest}>{children}</a>,
    ol: ({ children, ...rest}) => <ol className={classes.ol} {...rest}>{children}</ol>,
    ul: ({ children, ...rest}) => <ul className={classes.ul} {...rest}>{children}</ul>,
    li: ({ children, ...rest}) => <li className={classes.li} {...rest}>{children}</li>,
    code: ({ children, className, ...rest}) => <code className={classnames(className, classes.code)} {...rest}>{children}</code>,
    pre: ({ children, ...rest}) => <pre className={classes.pre} {...rest}>{children}</pre>,
    hr: ({ children, ...rest}) => <hr className={classes.hr} {...rest}>{children}</hr>,
    ...components,
  }
}