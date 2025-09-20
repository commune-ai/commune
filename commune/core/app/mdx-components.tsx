import type { MDXComponents } from 'mdx/types'
import { CopyButton } from './app/components/CopyButton'

export function useMDXComponents(components: MDXComponents): MDXComponents {
  return {
    code: (props) => (
      //   <div style={{ display: 'inline-block', borderRadius: 10, minWidth: '100%' }}>
      //     <div style={{ display: 'flex', padding: '16px', backgroundColor: '#0D0F14', borderRadius: '0px 10px 0px 0px' }}>
      //       <div style={{ width: '0.8rem', height: '0.8rem', backgroundColor: '#FF5F56', borderRadius: '100%', marginRight: 8 }} />
      //       <div style={{ width: '0.8rem', height: '0.8rem', backgroundColor: '#FFBD2D', borderRadius: '100%', marginRight: 8 }} />
      //       <div style={{ width: '0.8rem', height: '0.8rem', backgroundColor: '#26C940', borderRadius: '100%', marginRight: 8 }} />
      //     </div>
      //     <div
      //       style={{ width: '100%', backgroundColor: '#22212C', display: 'inline-flex', padding: '16px', borderRadius: '0px 0px 10px 0px' }}
      //     >
      <div className='flex items-center'>
        <div className='ml-20'>
          <code {...props}>{props.children}</code>
        </div>
        <CopyButton content={props.children as string} />
      </div>

      //     </div>
      //   </div>
      // </div>
    ),
    ...components,
  }
}
