import { ModuleType } from '@/app/types/module'
import { SearchFilters } from './AdvancedSearch'

/**
 * Apply advanced search filters to modules
 */
export function filterModules(modules: ModuleType[], filters: SearchFilters): ModuleType[] {
  return modules.filter(module => {
    // Basic search term matching - CASE INSENSITIVE
    if (filters.searchTerm) {
      const searchTerm = filters.searchTerm.toLowerCase()
      const matchesSearch = 
        module.name.toLowerCase().includes(searchTerm) ||
        (module.desc && module.desc.toLowerCase().includes(searchTerm)) ||
        (module.tags && module.tags.some(tag => tag.toLowerCase().includes(searchTerm)))
      
      if (!matchesSearch) return false
    }

    // Tag filtering - CASE INSENSITIVE
    if (module.tags) {
      const moduleTags = module.tags.map(tag => tag.toLowerCase())
      
      // Include tags - module must have ALL specified tags
      if (filters.includeTags.length > 0) {
        const hasAllIncludeTags = filters.includeTags.every(tag => 
          moduleTags.includes(tag.toLowerCase())
        )
        if (!hasAllIncludeTags) return false
      }
      
      // Exclude tags - module must not have ANY specified tags
      if (filters.excludeTags.length > 0) {
        const hasExcludeTag = filters.excludeTags.some(tag => 
          moduleTags.includes(tag.toLowerCase())
        )
        if (hasExcludeTag) return false
      }
    } else {
      // If module has no tags but we require include tags, exclude it
      if (filters.includeTags.length > 0) return false
    }

    // Term filtering in name and description - CASE INSENSITIVE
    const moduleText = `${module.name} ${module.desc || ''}`.toLowerCase()
    
    // Include terms - module must contain ALL specified terms
    if (filters.includeTerms.length > 0) {
      const hasAllIncludeTerms = filters.includeTerms.every(term => 
        moduleText.includes(term.toLowerCase())
      )
      if (!hasAllIncludeTerms) return false
    }
    
    // Exclude terms - module must not contain ANY specified terms
    if (filters.excludeTerms.length > 0) {
      const hasExcludeTerm = filters.excludeTerms.some(term => 
        moduleText.includes(term.toLowerCase())
      )
      if (hasExcludeTerm) return false
    }

    return true
  })
}

/**
 * Extract all unique tags from modules
 */
export function extractAllTags(modules: ModuleType[]): string[] {
  const tagSet = new Set<string>()
  
  modules.forEach(module => {
    if (module.tags) {
      module.tags.forEach(tag => {
        tagSet.add(tag) // Keep original case
      })
    }
  })
  
  return Array.from(tagSet).sort()
}

/**
 * Get tag frequency from modules
 */
export function getTagFrequency(modules: ModuleType[]): Map<string, number> {
  const tagFrequency = new Map<string, number>()
  
  modules.forEach(module => {
    if (module.tags) {
      module.tags.forEach(tag => {
        tagFrequency.set(tag, (tagFrequency.get(tag) || 0) + 1)
      })
    }
  })
  
  return tagFrequency
}

/**
 * Get most popular tags
 */
export function getPopularTags(modules: ModuleType[], limit: number = 10): string[] {
  const tagFrequency = getTagFrequency(modules)
  
  return Array.from(tagFrequency.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
    .map(([tag]) => tag)
}