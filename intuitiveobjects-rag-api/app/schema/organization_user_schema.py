from pydantic import BaseModel, root_validator


class LoginOrganizationUserSchema(BaseModel):
    email: str
    password: str


class CreateOrganizationUserSchema(BaseModel):
    email: str
    password: str
    category_id: str


class UpdateOrganizationUserSchema(BaseModel):
    email: str | None = None
    password: str | None = None
    category_id: str | None = None

    @root_validator(pre=True)
    def check_at_least_one_field(cls, values):
        if not any(values.values()):
            raise ValueError(
                "At least one field (email, category_id, or password) must be provided."
            )
        return values
