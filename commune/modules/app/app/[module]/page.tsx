import Module from '@/app/module/Module'

export default async function ModulesPage({ params }: { params: { module: string } }) {
  const moduleName = params.module
  const code = true
  const api = true
  return (
    <div className="flex justify-center w-full px-4 md:px-8">
      <div className="w-full max-w-7xl">
        <Module module_name={moduleName} code={code} api={api}/>
      </div>
    </div>
  )
}