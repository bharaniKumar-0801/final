from pydantic import BaseModel, EmailStr, Field, root_validator


class CreateOrganizationAdminSchema(BaseModel):
    name: str
    email: EmailStr
    password: str


class UpdateOrganizationAdminSchema(BaseModel):
    name: str | None = Field(None)
    email: EmailStr | None = Field(None)
    password: str | None = Field(None)

    @root_validator(pre=True)
    def check_at_least_one_field(cls, values):
        if not any(values.values()):
            raise ValueError(
                "At least one field (name, email, or password) must be provided."
            )
        return values


class LoginOrganizationAdminSchema(BaseModel):
    email: EmailStr
    password: str


class CreateCategorySchema(BaseModel):
    name: str


class UpdateCategorySchema(BaseModel):
    name: str

class CreateOrganizationAppConfigSchema(BaseModel):
    llm_model: str
    embedding_model: str
    temperature: float
    system_prompt: str
   
