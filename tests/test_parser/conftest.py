"""Shared fixtures for parser tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def sample_typescript_file(tmp_project: Path) -> Path:
    """Create a sample TypeScript file for testing."""
    code = """import { Request, Response } from "express";

interface User {
  id: number;
  name: string;
}

class UserService {
  private users: User[] = [];

  getUser(id: number): User | undefined {
    return this.users.find(u => u.id === id);
  }

  addUser(user: User): void {
    this.users.push(user);
  }
}

function createApp(): void {
  const service = new UserService();
  console.log("App started");
}
"""
    file_path = tmp_project / "app.ts"
    file_path.write_text(code)
    return file_path


@pytest.fixture
def sample_rust_file(tmp_project: Path) -> Path:
    """Create a sample Rust file for testing."""
    code = """use std::fmt;

pub trait Greetable {
    fn greet(&self) -> String;
}

pub struct Person {
    pub name: String,
    age: u32,
}

impl Person {
    pub fn new(name: &str, age: u32) -> Self {
        Person {
            name: name.to_string(),
            age,
        }
    }

    fn is_adult(&self) -> bool {
        self.age >= 18
    }
}

impl Greetable for Person {
    fn greet(&self) -> String {
        format!("Hello, I am {}", self.name)
    }
}

pub fn create_person(name: &str) -> Person {
    Person::new(name, 30)
}
"""
    file_path = tmp_project / "person.rs"
    file_path.write_text(code)
    return file_path


@pytest.fixture
def sample_complex_python(tmp_project: Path) -> Path:
    """Create a complex Python file with inheritance and imports."""
    code = '''"""Complex module for testing."""

import os
from pathlib import Path
from typing import List, Optional

MAX_SIZE: int = 1024


class BaseService:
    """Base class for services."""

    def __init__(self, name: str):
        self.name = name

    def _internal_method(self):
        pass

    def __repr__(self):
        return f"BaseService({self.name})"


class UserService(BaseService):
    """Service for user operations."""

    def __init__(self, name: str, db_url: str):
        super().__init__(name)
        self.db_url = db_url

    def get_user(self, user_id: int) -> Optional[dict]:
        """Get a user by ID."""
        return None

    async def fetch_users(self) -> List[dict]:
        """Async fetch all users."""
        return []


def create_service(name: str = "default") -> UserService:
    """Factory function."""
    return UserService(name, "sqlite:///db.sqlite")
'''
    file_path = tmp_project / "services.py"
    file_path.write_text(code)
    return file_path
