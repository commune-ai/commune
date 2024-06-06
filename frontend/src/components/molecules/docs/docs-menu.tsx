import classNames from "classnames";
import classes from './docs-menu.module.css';
import ActiveLink from "@/components/atoms/active-link";

const MENU_ITEMS = [{
    name: '👋 Introduction',
    url: '/docs/introduction'
}, {
    name: '🔽 Setup Commune',
    url: '/docs/setup-commune'
}, {
    name: 'Basics',
    url: '/docs/basics/create-commune',
    submenuItems: [{
        name: 'Create Commune',
        url: '/docs/basics/create-commune'
    }, {
        name: 'Deploy Module',
        url: '/docs/basics/deploy-module'
    }, {
        name: 'Register Commune',
        url: '/docs/basics/register-commune'
    }, {
        name: 'Namespace',
        url: '/docs/basics/namespace'
    }, {
        name: 'Connect a Module',
        url: '/docs/basics/connect-module'
    }, {
        name: 'PyParse Cli Basics',
        url: '/docs/basics/cli-basics'
    }, {
        name: 'Key Basics',
        url: '/docs/basics/key-basics'
    }],
}, {
    name: 'Modules',
    url: '/docs/modules/module-basics',
    submenuItems: [{
        name: 'Module Basics',
        url: '/docs/modules/module-basics'
    }, {
        name: 'Bittensor',
        url: '/docs/modules/bittensor'
    }, {
        name: 'Data-hf',
        url: '/docs/modules/data-hf'
    }, {
        name: 'Model Transformer',
        url: '/docs/modules/model-hf'
    }, {
        name: 'Data text realfakes',
        url: '/docs/modules/data-text-realfake'
    }, {
        name: 'Data text truthqa',
        url: '/docs/modules/data-text-truthqa'
    }, {
        name: 'Translate your site',
        url: '/docs/modules/translate-your-site'
    }, {
        name: 'Model Openai',
        url: '/docs/modules/model-openai'
    }, {
        name: 'Validator',
        url: '/docs/modules/validator'
    }, {
        name: 'Vali text realfake',
        url: '/docs/modules/vali-text-realfake'
    }],
}];

export default function DocsMenu() {
    return (
        <aside className='w-[20rem] flex-shrink-0 p-[20px] text-[18px] font-medium text-[#606770]'>
            <ol className="">
                {MENU_ITEMS.map(({
                    url,
                    name,
                    submenuItems,
                }) => (
                    <>
                        <li className="">
                            <ActiveLink
                                className={classNames(classes.menuLink, 'dark:text-gray-200')}
                                activeClassName={classes.active}
                                href={url}
                            >
                                {name}
                            </ActiveLink>
                        </li>
                        {submenuItems && (
                            <ol className={classes.list}>
                                {submenuItems.map(({
                                    url,
                                    name,
                                }) => (
                                    <li className={classes.listItem} key={name}>
                                        <ActiveLink
                                            className={classNames(classes.menuLink, 'dark:text-gray-400')}
                                            activeClassName={classes.active}
                                            href={url}
                                        >
                                            {name}
                                        </ActiveLink>
                                    </li>
                                ))}
                            </ol>
                        )}
                    </>))}
            </ol>
        </aside>
    );
}