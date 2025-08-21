
import Module from '@/app/module/Module'
// use the [module] from the path as the argument to the Module component

export default async function ModulesPage({ params }: { params: { module: string } }) {

  const moduleName = params.module
  const code = true
  const api = true
  return (
        <Module module_name={moduleName} code={code} api={api}/>
  )
}