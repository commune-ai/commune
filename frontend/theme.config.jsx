export default {
    project: {
      link: 'https://github.com/commune-ai',
    },
    logo: <strong>Commune</strong>,
    navigation: {
        prev: true,
        next: true
      },
    sidebar : {
      titleComponent({ title, type }) {
        if (type === 'separator') {
          return (
            <div style={{ background: 'cyan', textAlign: 'center' }}>{title}</div>
          )
        }
        if (title === 'About') {
          return <>{title}</>
        }
        return <>{title}</>
      }
    },
    nextThemes: {
        forcedTheme : "dark",
      },
    darkMode : false
  }