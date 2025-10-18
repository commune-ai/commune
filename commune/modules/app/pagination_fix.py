#!/usr/bin/env python3
"""
Pagination Fix Module

This module provides a comprehensive pagination solution that can be integrated
into any Python application. It handles common pagination scenarios including:
- Page-based pagination
- Offset-based pagination
- Cursor-based pagination
- API response formatting
"""

import math
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class PaginationParams:
    """Parameters for pagination requests"""
    page: int = 1
    page_size: int = 10
    offset: Optional[int] = None
    cursor: Optional[str] = None
    sort_by: Optional[str] = None
    sort_order: str = 'asc'
    
    def __post_init__(self):
        # Validate parameters
        if self.page < 1:
            self.page = 1
        if self.page_size < 1:
            self.page_size = 10
        if self.page_size > 100:
            self.page_size = 100
        if self.offset is None:
            self.offset = (self.page - 1) * self.page_size


@dataclass
class PaginationResponse:
    """Standard pagination response structure"""
    data: List[Any]
    total_items: int
    total_pages: int
    current_page: int
    page_size: int
    has_next: bool
    has_previous: bool
    next_page: Optional[int]
    previous_page: Optional[int]
    next_cursor: Optional[str] = None
    previous_cursor: Optional[str] = None


class PaginationStrategy(ABC):
    """Abstract base class for pagination strategies"""
    
    @abstractmethod
    def paginate(self, items: List[Any], params: PaginationParams) -> PaginationResponse:
        """Apply pagination to a list of items"""
        pass


class OffsetPagination(PaginationStrategy):
    """Offset-based pagination strategy"""
    
    def paginate(self, items: List[Any], params: PaginationParams) -> PaginationResponse:
        total_items = len(items)
        total_pages = math.ceil(total_items / params.page_size)
        
        # Apply sorting if specified
        if params.sort_by:
            reverse = params.sort_order.lower() == 'desc'
            try:
                items = sorted(items, key=lambda x: getattr(x, params.sort_by, None), reverse=reverse)
            except:
                # If sorting fails, continue with unsorted items
                pass
        
        # Calculate slice boundaries
        start_idx = params.offset
        end_idx = start_idx + params.page_size
        
        # Slice the items
        paginated_items = items[start_idx:end_idx]
        
        # Determine navigation properties
        has_next = params.page < total_pages
        has_previous = params.page > 1
        next_page = params.page + 1 if has_next else None
        previous_page = params.page - 1 if has_previous else None
        
        return PaginationResponse(
            data=paginated_items,
            total_items=total_items,
            total_pages=total_pages,
            current_page=params.page,
            page_size=params.page_size,
            has_next=has_next,
            has_previous=has_previous,
            next_page=next_page,
            previous_page=previous_page
        )


class CursorPagination(PaginationStrategy):
    """Cursor-based pagination strategy for large datasets"""
    
    def __init__(self, cursor_field: str = 'id'):
        self.cursor_field = cursor_field
    
    def paginate(self, items: List[Any], params: PaginationParams) -> PaginationResponse:
        # Sort items by cursor field
        try:
            items = sorted(items, key=lambda x: getattr(x, self.cursor_field, 0))
        except:
            items = sorted(items, key=lambda x: x.get(self.cursor_field, 0) if isinstance(x, dict) else 0)
        
        # Find starting position based on cursor
        start_idx = 0
        if params.cursor:
            for i, item in enumerate(items):
                item_cursor = getattr(item, self.cursor_field, None) if hasattr(item, self.cursor_field) else item.get(self.cursor_field)
                if str(item_cursor) == params.cursor:
                    start_idx = i + 1
                    break
        
        # Get page of items
        end_idx = start_idx + params.page_size
        paginated_items = items[start_idx:end_idx]
        
        # Determine cursors
        next_cursor = None
        previous_cursor = None
        
        if paginated_items:
            if end_idx < len(items):
                last_item = paginated_items[-1]
                next_cursor = str(getattr(last_item, self.cursor_field, None) if hasattr(last_item, self.cursor_field) else last_item.get(self.cursor_field))
            
            if start_idx > 0 and start_idx - 1 < len(items):
                prev_item = items[start_idx - 1]
                previous_cursor = str(getattr(prev_item, self.cursor_field, None) if hasattr(prev_item, self.cursor_field) else prev_item.get(self.cursor_field))
        
        total_items = len(items)
        total_pages = math.ceil(total_items / params.page_size)
        current_page = math.ceil((start_idx + 1) / params.page_size)
        
        return PaginationResponse(
            data=paginated_items,
            total_items=total_items,
            total_pages=total_pages,
            current_page=current_page,
            page_size=params.page_size,
            has_next=end_idx < len(items),
            has_previous=start_idx > 0,
            next_page=current_page + 1 if end_idx < len(items) else None,
            previous_page=current_page - 1 if start_idx > 0 else None,
            next_cursor=next_cursor,
            previous_cursor=previous_cursor
        )


class Paginator:
    """Main paginator class that manages pagination strategies"""
    
    def __init__(self, strategy: PaginationStrategy = None):
        self.strategy = strategy or OffsetPagination()
    
    def paginate(self, items: List[Any], page: int = 1, page_size: int = 10, **kwargs) -> PaginationResponse:
        """Paginate a list of items"""
        params = PaginationParams(page=page, page_size=page_size, **kwargs)
        return self.strategy.paginate(items, params)
    
    def paginate_dict(self, items: List[Any], page: int = 1, page_size: int = 10, **kwargs) -> Dict[str, Any]:
        """Paginate and return as dictionary (useful for JSON APIs)"""
        response = self.paginate(items, page, page_size, **kwargs)
        return {
            'data': response.data,
            'pagination': {
                'total_items': response.total_items,
                'total_pages': response.total_pages,
                'current_page': response.current_page,
                'page_size': response.page_size,
                'has_next': response.has_next,
                'has_previous': response.has_previous,
                'next_page': response.next_page,
                'previous_page': response.previous_page,
                'next_cursor': response.next_cursor,
                'previous_cursor': response.previous_cursor
            }
        }


# Utility functions for common pagination scenarios

def paginate_query(query, page: int = 1, page_size: int = 10) -> Tuple[Any, Dict[str, Any]]:
    """Paginate a database query (SQLAlchemy style)"""
    total_items = query.count()
    total_pages = math.ceil(total_items / page_size)
    offset = (page - 1) * page_size
    
    items = query.limit(page_size).offset(offset).all()
    
    pagination_info = {
        'total_items': total_items,
        'total_pages': total_pages,
        'current_page': page,
        'page_size': page_size,
        'has_next': page < total_pages,
        'has_previous': page > 1,
        'next_page': page + 1 if page < total_pages else None,
        'previous_page': page - 1 if page > 1 else None
    }
    
    return items, pagination_info


def get_pagination_links(base_url: str, current_page: int, total_pages: int) -> Dict[str, str]:
    """Generate pagination links for REST APIs"""
    links = {
        'self': f"{base_url}?page={current_page}",
        'first': f"{base_url}?page=1",
        'last': f"{base_url}?page={total_pages}"
    }
    
    if current_page > 1:
        links['previous'] = f"{base_url}?page={current_page - 1}"
    
    if current_page < total_pages:
        links['next'] = f"{base_url}?page={current_page + 1}"
    
    return links


# Example usage and integration helpers

class PaginationMixin:
    """Mixin for adding pagination to class-based views"""
    
    paginator_class = Paginator
    page_size = 10
    
    def get_paginated_response(self, items: List[Any], request_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get paginated response for API views"""
        page = int(request_params.get('page', 1))
        page_size = int(request_params.get('page_size', self.page_size))
        
        paginator = self.paginator_class()
        return paginator.paginate_dict(items, page=page, page_size=page_size)


# Flask/FastAPI integration example
def create_pagination_params(request) -> PaginationParams:
    """Extract pagination parameters from request"""
    return PaginationParams(
        page=int(request.args.get('page', 1)),
        page_size=int(request.args.get('page_size', 10)),
        sort_by=request.args.get('sort_by'),
        sort_order=request.args.get('sort_order', 'asc'),
        cursor=request.args.get('cursor')
    )


if __name__ == '__main__':
    # Example usage
    print("Pagination Fix Module - Examples\n")
    
    # Create sample data
    sample_data = [{'id': i, 'name': f'Item {i}'} for i in range(1, 101)]
    
    # Example 1: Basic offset pagination
    print("1. Offset Pagination Example:")
    paginator = Paginator()
    result = paginator.paginate_dict(sample_data, page=2, page_size=10)
    print(f"Page 2 of {result['pagination']['total_pages']}")
    print(f"Items: {result['data'][:3]}...")
    print(f"Has next: {result['pagination']['has_next']}")
    print(f"Has previous: {result['pagination']['has_previous']}\n")
    
    # Example 2: Cursor pagination
    print("2. Cursor Pagination Example:")
    cursor_paginator = Paginator(CursorPagination(cursor_field='id'))
    cursor_result = cursor_paginator.paginate_dict(sample_data, page_size=10, cursor='10')
    print(f"Items after cursor '10': {cursor_result['data'][:3]}...")
    print(f"Next cursor: {cursor_result['pagination']['next_cursor']}")
    print(f"Previous cursor: {cursor_result['pagination']['previous_cursor']}\n")
    
    # Example 3: Pagination links
    print("3. Pagination Links Example:")
    links = get_pagination_links('https://api.example.com/items', current_page=5, total_pages=10)
    for rel, url in links.items():
        print(f"{rel}: {url}")
    
    print("\nPagination fix module ready for integration!")
