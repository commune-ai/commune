
import ModulesClient from '@/app/module/page/ModulePage'

export default async function ModulesPage({ params, searchParams }: { params : { module: string }, searchParams: { [key: string]: string | string[] | undefined } }) {
  const module = params.module
  const code = searchParams.code === 'true'
  const api = searchParams.api === 'true'
  return (
        <ModulesClient module_name={module} code={code} api={api}/>
  )
}