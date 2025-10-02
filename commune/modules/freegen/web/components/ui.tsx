import { clsx } from 'clsx';
import { ComponentProps, PropsWithChildren } from 'react';
export function Button({ variant = 'primary', className, children, ...rest }: PropsWithChildren<{ variant?: 'primary'|'outline'|'ghost' }> & ComponentProps<'button'>) {
  const base = 'btn';
  const map = { primary: 'btn-primary', outline: '', ghost: '' };
  return <button className={clsx(base, map[variant], className)} {...rest}>{children}</button>;
}
