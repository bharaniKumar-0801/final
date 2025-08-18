from pydantic import BaseModel


class CreateOrganizationSchema(BaseModel):
    name: str


class UpdateOrganizationSchema(BaseModel):
    name: str
